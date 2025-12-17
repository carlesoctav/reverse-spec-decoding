
from dataclasses import dataclass, asdict
from vllm.sampling_params import SamplingParams, RequestOutputKind
from vllm import AsyncLLMEngine, LLM
from datasets import load_dataset, concatenate_datasets
from functools import partial

@dataclass
class SamplingParamsArgs:
    max_tokens: int = 1
    temperature: float = 0.0
    logprobs: int = 20

    def make(self) -> SamplingParams:
        pass

@dataclass
class HFSource:
    path: str
    name: str | None = None
    split: str | None = None
    num_proc: int = 8
    prompt_column: str
    ground_truth_column: str

    def as_dict(self):
        return {
            "path":self.path,
            "name":self.name,
            "split":self.split,
            "num_proc":self.num_proc,
        }


@dataclass
class Dataset:
    sources: list[HFSource]
    batch: int
    num_proc: int

    def make(self):
        ds = []
        for source in self.sources:
            hf_ds = load_dataset(source.as_dict())
            ds.append(hf_ds)
            ds = ds.map(partial(self.prepare, ), remove_columns = ds.column_names)
        ds  = concatenate_datasets(ds)


@dataclass
class Args:
    student_model: str
    teacher_model: str
    custom_teacher_template: Path | str
    custom_student_template: Path | str
    dataset: List[HF]


def mix_decode():
    sampling_params = SamplingParamsArgs()
    pass



def main():
    pass


if __name__ == "__main__" :
    main()
