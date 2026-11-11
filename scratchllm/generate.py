import argparse

import torch
from tokenizers import Tokenizer

from .model import GPT, ModelConfig
from .links import strip_links
from .dialogue import CHAT_FORMAT, prompt_tokens


class Generator:
    def __init__(self, checkpoint, device="cuda"):
        self.device = torch.device(device)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model = GPT(ModelConfig(**state["model_config"]))
        self.model.load_state_dict(state["model"])
        self.model.to(self.device).eval()
        self.tokenizer = Tokenizer.from_str(state["tokenizer_json"])
        self.chat_format = state.get('chat_format')

    def complete(self, prompt, max_new_tokens=256, temperature=0.7, top_k=40, top_p=0.9, repetition_penalty=1.1):
        tokens = [self.tokenizer.token_to_id("<|bos|>"), *self.tokenizer.encode(prompt).ids]
        tokens = tokens[-self.model.cfg.context:]
        result = self._generate(tokens, max_new_tokens, temperature, top_k, top_p, repetition_penalty)
        return dict(result, text=prompt + result['completion'])

    def chat(self, prompt, history=None, max_new_tokens=256, temperature=0.7, top_k=40, top_p=0.9, repetition_penalty=1.1,
             collect_stats=False):
        if self.chat_format != CHAT_FORMAT:
            raise ValueError('Load a dialogue-fine-tuned checkpoint to use chat mode')
        messages = [*(history or []), dict(role='user', content=prompt)]
        while True:
            tokens = prompt_tokens(self.tokenizer, messages)
            if len(tokens) <= self.model.cfg.context:
                break
            if len(messages) == 1:
                raise ValueError('Message exceeds the model context; shorten it')
            messages = messages[2:]
        options = {'collect_stats': True} if collect_stats else {}
        result = self._generate(tokens, max_new_tokens, temperature, top_k, top_p, repetition_penalty, **options)
        # A well-fitted model emits EOS. Also avoid displaying a generated second speaker.
        completion = result['completion'].split('\nUser:')[0].split('\nAssistant:')[0].strip()
        return dict(result, completion=completion, text=completion, history_messages_used=len(messages) - 1)

    def _generate(self, tokens, max_new_tokens, temperature, top_k, top_p, repetition_penalty, collect_stats=False):
        ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        forbidden = tuple(self.tokenizer.token_to_id(token) for token in ('<|pad|>', '<|bos|>'))
        options = {'collect_stats': True} if collect_stats else {}
        result = self.model.generate(ids, max_new_tokens, temperature, top_k, self.tokenizer.token_to_id("<|eos|>"),
                                     top_p=top_p, repetition_penalty=repetition_penalty, forbidden_ids=forbidden, **options)
        diagnostics = {}
        if collect_stats:
            result, diagnostics = result
        continuation = result[0, len(tokens):].tolist()
        decoded = self.tokenizer.decode(continuation)
        completion = strip_links(decoded)
        # Retain the separating whitespace of an ordinary continuation.
        if decoded[:1].isspace() and completion:
            completion = decoded[:1] + completion
        return {"completion": completion,
                "prompt_tokens_used": len(tokens), "generated_tokens": len(continuation),
                **({'diagnostics': diagnostics} if collect_stats else {})}


def main():
    p = argparse.ArgumentParser(description="Sample your locally trained model")
    p.add_argument("--checkpoint", default="runs/tiny/best.pt")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--prompt", default="The")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    p.add_argument('--chat', action='store_true', help='Use the dialogue checkpoint prompt format')
    args = p.parse_args()
    generator = Generator(args.checkpoint, args.device)
    method = generator.chat if args.chat else generator.complete
    print(method(args.prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k,
                 top_p=args.top_p, repetition_penalty=args.repetition_penalty)['text'])


if __name__ == "__main__":
    main()
