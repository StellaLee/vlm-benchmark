"""Self-reflection / abstention strategies (O2-KR1).

`self_reflection` is a two-turn prompt: the model answers, then is asked to
critique its own answer against the visual evidence and produce a (possibly
revised) final answer + confidence, with an explicit option to abstain.

`abstention` is a single-turn variant that simply licenses the model to decline
when the visual input is insufficient ("I cannot determine ...").

Both elicit `verbal_confidence`; abstention is detected via parsing.is_abstention
and recorded on the Prediction. self_reflection costs 2 API calls per sample.
"""

from avbench.inference.client import VLMClient
from avbench.inference.parsing import extract_answer, extract_confidence, is_abstention
from avbench.inference.strategies.base import PromptStrategy, register, render_question
from avbench.schema import Prediction, Sample

_ABSTAIN_OPTION = (
    'If the relevant camera view does not contain enough information to answer '
    'confidently, respond exactly: "I cannot determine this from the available '
    'visual input."'
)

_ANSWER_FORMAT = (
    "Respond in exactly this format:\n"
    "Answer: <your answer>\n"
    "Confidence: <0-100>"
)

_REFLECT_TEMPLATE = (
    "{body}\n\n"
    "Your initial answer was:\n\"\"\"\n{first}\n\"\"\"\n\n"
    "Now review that answer critically:\n"
    "- What visual evidence in the images supports it?\n"
    "- Is the relevant camera view clear enough to be certain?\n"
    "- Do you want to revise the answer or your confidence?\n"
    "{abstain}\n\n"
    "Then give your FINAL answer and confidence.\n{fmt}"
)


@register("self_reflection")
class SelfReflection(PromptStrategy):
    def build_prompt(self, sample: Sample) -> str:
        # Turn 1: a plain answer we will later ask the model to reflect on.
        how = ("Answer with the single best option letter." if sample.options
               else "Answer concisely.")
        return "{}\n\n{}\nAnswer:".format(render_question(sample), how)

    def _reflect_prompt(self, sample: Sample, first_text: str) -> str:
        return _REFLECT_TEMPLATE.format(
            body=render_question(sample),
            first=first_text.strip(),
            abstain=_ABSTAIN_OPTION,
            fmt=_ANSWER_FORMAT,
        )

    async def run(self, sample: Sample, client: VLMClient) -> Prediction:
        first = (await client.generate(self.build_prompt(sample), sample.images, n=1))[0]
        reflect = self._reflect_prompt(sample, first.text)
        final = (await client.generate(reflect, sample.images, n=1))[0]
        return Prediction(
            sample_id=sample.sample_id,
            model=client.model,
            strategy=self.name,
            raw_text=final.text,
            answer=extract_answer(final.text, sample),
            verbal_confidence=extract_confidence(final.text),
            abstained=is_abstention(final.text),
            samples=[first.text, final.text],  # turn-1 and turn-2 transcripts
            usage={"calls": 2, "backend": final.usage.get("backend")},
        )


@register("abstention")
class Abstention(PromptStrategy):
    def build_prompt(self, sample: Sample) -> str:
        how = ("Choose the single best option and give its letter." if sample.options
               else "Give a concise answer.")
        return (
            "{body}\n\n"
            "Before answering, assess whether the images give you enough "
            "information to answer confidently. {abstain}\n"
            "Otherwise, {how}\n\n{fmt}"
        ).format(body=render_question(sample), abstain=_ABSTAIN_OPTION, how=how, fmt=_ANSWER_FORMAT)

    async def run(self, sample: Sample, client: VLMClient) -> Prediction:
        res = (await client.generate(self.build_prompt(sample), sample.images, n=1))[0]
        return Prediction(
            sample_id=sample.sample_id,
            model=client.model,
            strategy=self.name,
            raw_text=res.text,
            answer=extract_answer(res.text, sample),
            verbal_confidence=extract_confidence(res.text),
            abstained=is_abstention(res.text),
            usage=res.usage,
        )
