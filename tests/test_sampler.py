from dataclasses import dataclass

from stage_state_vla.trainer import EqualTaskSampler


@dataclass
class Row:
    task: str


class Data:
    records = [Row("large")] * 100 + [Row("small")] * 2

    def __len__(self):
        return len(self.records)


def test_equal_task_sampler_does_not_follow_frame_count():
    sampled = list(EqualTaskSampler(Data(), seed=7, num_samples=10000))
    small = sum(index >= 100 for index in sampled)
    assert 0.47 < small / len(sampled) < 0.53
