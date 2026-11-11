"""Bounded retrieve → draft → optional review → remember loop for local chat."""
import re

from .dialogue import CHAT_FORMAT, message_text, prompt_tokens


def assemble(generator, question, history, notes, max_new_tokens):
    """Budget actual tokenizer IDs; keep the full current question and complete old turns."""
    tokenizer, context = generator.tokenizer, generator.model.cfg.context
    def size(text, turns=()):
        return len(prompt_tokens(tokenizer, [*turns, dict(role='user', content=text)]))
    base = size(question)
    if base > context:
        raise ValueError('Message exceeds the model context; shorten it')
    limit = max(base, context - min(max_new_tokens, context // 4))
    selected, used = [], []
    def render(extra):
        if not extra:
            return question
        instruction = 'Review the draft and answer the current question:' if extra[0].startswith('Draft to review: ') else 'Current question:'
        return 'Context notes (recorded, not verified):\n' + '\n'.join(extra) + '\n\n' + instruction + '\n' + question
    for key, text in notes:
        ids = tokenizer.encode(message_text(text)).ids[:max(16, context // 6)]
        low, high = 0, len(ids)
        while low < high:
            mid = (low + high + 1) // 2
            candidate = render([*selected, tokenizer.decode(ids[:mid])])
            if size(candidate) <= min(limit, base + context // 3):
                low = mid
            else:
                high = mid - 1
        excerpt = tokenizer.decode(ids[:low])
        if key == 'draft' and (not excerpt.startswith('Draft to review: ') or len(excerpt) <= len('Draft to review: ') + 3):
            continue
        if low >= min(16, len(ids)) and low:
            selected.append(excerpt)
            used.append(key)
    text = render(selected)
    turns = list(history)
    while turns and size(text, turns) > limit:
        turns = turns[2:]
    return text, turns, used


def self_model(generator, checkpoint, snapshot, goal):
    return dict(checkpoint=checkpoint, architecture=generator.model.config_dict(), goal=goal,
                memory='session-scoped saved exchanges with lexical retrieval', feedback=snapshot['feedback'],
                capabilities=['text generation', 'saved conversation retrieval', 'optional single draft review'],
                tools=[], factual_confidence_calibrated=False)


def chat(generator, store, checkpoint, session_id, prompt, goal=None, review=False, **options):
    if generator.chat_format != CHAT_FORMAT:
        raise ValueError('Load a dialogue-fine-tuned checkpoint to use memory chat')
    prompt = message_text(prompt)
    if not prompt:
        raise ValueError('Enter a nonempty message')
    snapshot = store.snapshot(session_id)
    goal = snapshot['goal'] if goal is None else message_text(goal)
    recent = [row for row in snapshot['recent'] if row['feedback'] != -1][-3:]
    history = [dict(role=role, content=row[key]) for row in recent
               for role, key in [('user', 'user_text'), ('assistant', 'assistant_text')]]
    memories = store.retrieve(session_id, prompt, exclude=tuple(row['id'] for row in recent))
    notes = ([('goal', 'User goal: ' + goal)] if goal else [])
    notes.append(('self', f'Local text assistant. Context: {generator.model.cfg.context} tokens. No external tools.'))
    notes.extend((row['id'], f"Recorded #{row['id']} — User: {row['user_text'][:320]} Assistant: {row['assistant_text'][:320]}") for row in memories)
    text, turns, used = assemble(generator, prompt, history, notes, options.get('max_new_tokens', 256))
    result = generator.chat(text, history=turns, collect_stats=True, **options)
    total_tokens = result['generated_tokens']
    review_state = dict(requested=review, performed=False)
    if review and result['completion']:
        review_notes = [('draft', 'Draft to review: ' + result['completion']), *notes]
        revised_text, revised_turns, revised_used = assemble(generator, prompt, history, review_notes, options.get('max_new_tokens', 256))
        if 'draft' in revised_used:
            revised = generator.chat(revised_text, history=revised_turns, collect_stats=True, **options)
            total_tokens += revised['generated_tokens']
            review_state.update(performed=True, draft_diagnostics=result.get('diagnostics', {}),
                                used_revised_answer=bool(revised['completion']))
            if revised['completion']:
                result, used, turns = revised, revised_used, revised_turns
        else:
            review_state['reason'] = 'No context space for a draft'
    words = re.findall(r'\w+', result['completion'].casefold())
    grams = [tuple(words[i:i + 4]) for i in range(max(0, len(words) - 3))]
    diagnostics = dict(result.get('diagnostics', {}), empty_response=not bool(result['completion']),
                       repeated_fourgram_fraction=1 - len(set(grams)) / len(grams) if grams else 0,
                       review=review_state)
    episode_id = None
    if result['completion']:
        episode_id = store.record(session_id, prompt, result['completion'], checkpoint, diagnostics, goal, snapshot['revision'])
    model = self_model(generator, checkpoint, snapshot, goal)
    return dict(result, diagnostics=diagnostics, total_generated_tokens=total_tokens,
                workspace=dict(session_id=session_id, episode_id=episode_id, revision=snapshot['revision'] + bool(episode_id),
                               goal=goal, self_model=model, context_notes_used=used,
                               memory_ids=[key for key in used if isinstance(key, int)],
                               recent_episode_ids=[row['id'] for row in recent[-(len(turns)//2):]] if turns else [],
                               review=review_state))
