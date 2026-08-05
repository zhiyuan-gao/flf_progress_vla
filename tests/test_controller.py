from stage_state_vla.controller import MonotonicStageController


def test_controller_requires_confirmation_and_never_skips():
    controller = MonotonicStageController(3, completion_threshold=0.9, confirmations=2)
    first = controller.update(0.95)
    assert first.stage_index == 0 and not first.advanced
    second = controller.update(0.92)
    assert second.stage_index == 1 and second.advanced and second.progress == 0.0
    assert controller.update(0.99).stage_index == 1
    assert controller.update(0.99).stage_index == 2
    assert controller.update(0.99).stage_index == 2
    terminal = controller.update(0.99)
    assert terminal.stage_index == 2 and terminal.terminal
