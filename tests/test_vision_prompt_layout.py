from pacman_recipe.level1.prompts import (
    EDWARD_OPTION_CODE_V2_USER_TEMPLATE, build_image_messages,
    layout_image_user_content, live_state_instruction,
    VISION_EDWARD_FIXED_PREFIX, render_vision_edward_decision_prompt,
)
from pacman_recipe.level1.vision_prompt import grounding_messages


def test_edward_fixed_prefix_precedes_image_and_dynamic_state():
    from types import SimpleNamespace
    prefix = VISION_EDWARD_FIXED_PREFIX
    instruction = render_vision_edward_decision_prompt(
        {'pacman_position':[1,2]}, [], SimpleNamespace(rendered_choices=['B']))
    content = layout_image_user_content(
        b"png", instruction, prompt_style="live_state_v3", edward_options=True)
    assert [x["type"] for x in content] == ["text", "image_url", "text"]
    assert content[0]["text"] == prefix
    assert content[2]["text"].startswith('[CURRENT STATE]\n{"p"')
    assert '"c":' not in content[2]["text"]
    assert '[CANDIDATE OBJECTIVES]' in content[2]['text']
    assert '[OUTPUT]' in content[2]['text']
    assert "".join(x["text"] for x in content if x["type"] == "text") == instruction


def test_live_state_layout_preserves_dynamic_state_and_action_constraint():
    state = dict(pacman_position=[1, 2], open_actions=["U", "R"])
    content = build_image_messages(b"png", prompt_style="live_state_v3",
                                   state_context=state)[1]["content"]
    assert [x["type"] for x in content] == ["text", "image_url", "text"]
    assert "GAME STATE" not in content[0]["text"]
    assert "row=1, col=2" in content[2]["text"]
    assert content[2]["text"].endswith("Choose ONE ACTION from [U, R].")
    assert content[0]["text"] + content[2]["text"] == live_state_instruction(state)


def test_grounding_layout_same_image_prefix_for_different_questions():
    first = grounding_messages("rules", "Where is Pacman?", fixed_text="Read the image.\n")
    second = grounding_messages("rules", "Where is the ghost?", fixed_text="Read the image.\n")
    assert first[1]["content"][:2] == second[1]["content"][:2]
    assert [x["type"] for x in first[1]["content"]] == ["text", "image", "text"]
    assert grounding_messages("rules", "Question", fixed_text='')[1]["content"][0] == {"type": "image"}


def test_grounding_default_visible_only_and_task_specific_json():
    from pacman_recipe.level1.vision_prompt import GROUNDING_SYSTEM
    question='Read these cells in order: [[0,0],[1,2]].'
    answer_format='Output exactly {"symbols":[2 one-character strings]}.'
    train=grounding_messages(None,question,answer_format=answer_format)
    evaluation=grounding_messages(None,question,answer_format=answer_format)
    assert train==evaluation
    assert train[0]['content']==GROUNDING_SYSTEM
    fixed,image,dynamic=train[1]['content']
    assert image=={'type':'image'}
    assert '[row,column]' in fixed['text'] and 'starting at 0' in fixed['text']
    assert 'board cells, not pixels' in fixed['text']
    assert 'hidden' in fixed['text'] and 'never invent a position' in fixed['text']
    assert dynamic['text']==f'[QUESTION]\n{question}\n\n[OUTPUT]\n{answer_format}'


def test_visual_sections_preserve_candidates_and_risk_evidence():
    import json
    from types import SimpleNamespace
    from pacman_recipe.level1.prompts import render_edward_decision_prompt
    candidate=SimpleNamespace(option_id='A0',strategy='RISK_FALLBACK',target=(1,2),
        first_action='R',route_distance=1,commit_moves=1,safety_margin=0,
        future_safe_exits=1,entity_id=None,risk=dict(rank=1,motion='unknown',
        ghost_clearance=2,route_margin=0,safe_next_cells=1,dead_end=False,reverse=False))
    constraint=SimpleNamespace(rendered_choices=['B'],code_for_option=lambda _: 'B')
    old=render_edward_decision_prompt({},[candidate],constraint,fallback_mode='risk_ranked')
    new=render_vision_edward_decision_prompt({},[candidate],constraint,fallback_mode='risk_ranked')
    state=json.loads(new.split('[CURRENT STATE]\n')[1].splitlines()[0])
    row=json.loads(new.split('[CANDIDATE OBJECTIVES]\n')[1].splitlines()[0])
    previous=json.loads(old.splitlines()[2])
    assert state=={k:v for k,v in previous.items() if k!='c'}
    assert row==previous['c'][0]
    marker=' RISK_FALLBACK is not safety-approved.'
    assert new[new.index(marker):]==old[old.index(marker):]
    from pacman_recipe.level1.prompts import compact_ascii_edward_decision_prompt
    ascii_prompt=compact_ascii_edward_decision_prompt(
        {},[candidate],constraint,
        'MAP (row 0 at top; header shows column mod 10):\n  012\n0 #P ',
        fallback_mode='risk_ranked')
    ascii_state=json.loads(ascii_prompt.split('[CURRENT STATE]\n')[1].splitlines()[0])
    ascii_row=json.loads(ascii_prompt.split('[CANDIDATE OBJECTIVES]\n')[1].splitlines()[0])
    assert ascii_state==state and ascii_row==row
    assert ascii_prompt.split('[OUTPUT]\n')[1]==new.split('[OUTPUT]\n')[1]


def test_native_transport_preserves_fixed_image_dynamic_order():
    import numpy as np
    from pacman_recipe.level1.image_transport import pil_and_chat_messages
    from pacman_recipe.level1.prompts import encode_png
    pixels = np.zeros((16, 16, 3), dtype=np.uint8)
    messages = build_image_messages(encode_png(pixels), prompt_style="live_state_v3",
                                    state_context=dict(pacman_position=[1, 2]))
    image, chat = pil_and_chat_messages(messages)
    assert [x["type"] for x in chat[1]["content"]] == ["text", "image", "text"]
    assert chat[1]["content"][1]["image"] is image
    np.testing.assert_array_equal(np.asarray(image), pixels)
