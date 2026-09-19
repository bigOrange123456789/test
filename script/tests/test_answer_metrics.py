"""客观答案解析与准确率测试；包括本地 MIRA 结果中的实际回答格式。"""

import json
from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from script.lib.answer_metrics import score_answer


def choice_sample(*, multiple=False, reference=None, options=None):
    category = "multiple_choice" if multiple else "single_choice"
    return {
        "id": f"mira:train:0:{category}:0", "question_type": category,
        "question": "Which option is correct?", "images": [],
        "options": options or ["A. MRI", "B. CT", "C. Echocardiography", "D. X-ray"],
        "answer": reference if reference is not None else
                  ({"correct_options": ["A", "C"]} if multiple else {"correct_option": "C. Echocardiography"}),
    }


class AnswerMetricTests(unittest.TestCase):
    def test_real_single_choice_json_and_direct_text_have_same_normalized_answer(self):
        sample = choice_sample()
        for prediction in ("C", "C.", "C. Echocardiography", "Echocardiography", "C (Echocardiography)",
                           "Answer: C", "Final answer: C.", "**C. Echocardiography**",
                           json.dumps({"correct_option": "C. Echocardiography", "explanation": "wrong explanation"}),
                           '```json\n{"correct_option":"C"}\n```'):
            with self.subTest(prediction=prediction):
                result = score_answer(sample, prediction)
                self.assertEqual(result["answer_accuracy"], 1.0)
                self.assertTrue(result["eligible"])
                self.assertEqual(result["normalized_prediction"], "C")
                self.assertEqual(result["normalized_reference"], "C")
                self.assertEqual(result["parse_status"], "ok")

    def test_incorrect_single_choice_keeps_valid_parse_and_zero_accuracy(self):
        result = score_answer(choice_sample(), '{"correct_option":"B. CT"}')
        self.assertEqual(result["answer_accuracy"], 0.0)
        self.assertEqual(result["parse_status"], "ok")
        self.assertEqual(result["normalized_prediction"], "B")

    def test_negation_reasoning_and_ambiguous_choices_are_never_harvested(self):
        for prediction in ("not C", "C is not correct", "The answer is not C", "I considered C but rejected it.",
                           "C or B", "C, B", "C. Not echocardiography", "C. CT", "Maybe C", "Z", "",
                           '{"explanation":"The answer might be C"}', '{"correct_option":"C",'):
            with self.subTest(prediction=prediction):
                result = score_answer(choice_sample(), prediction)
                self.assertEqual(result["answer_accuracy"], 0.0)
                self.assertTrue(result["eligible"], "解析失败仍计入客观题分母")
                self.assertEqual(result["parse_status"], "prediction_unparseable")

    def test_explicit_final_answer_can_follow_reasoning_without_collecting_its_options(self):
        prediction = "A and B were discussed but rejected.\n\nFinal answer: C.\n\nExplanation: D is wrong."
        result = score_answer(choice_sample(), prediction)
        self.assertEqual(result["answer_accuracy"], 1.0)
        self.assertEqual(result["normalized_prediction"], "C")
        self.assertEqual(result["prediction_raw"], prediction)
        conflicting = score_answer(choice_sample(), "Answer: A\n\nFinal answer: C")
        self.assertEqual(conflicting["parse_status"], "prediction_unparseable")

    def test_multiple_choice_order_does_not_matter_but_over_and_under_selection_fail(self):
        sample = choice_sample(multiple=True)
        for prediction in ('{"correct_options":["C", "A"]}', "A, C", "C and A", "A和C", "A、C",
                           "Final answer: A; C", '{"correct_options":["MRI","C. Echocardiography"]}'):
            with self.subTest(prediction=prediction):
                result = score_answer(sample, prediction)
                self.assertEqual(result["answer_accuracy"], 1.0)
                self.assertEqual(result["choice_f1"], 1.0)
                self.assertEqual(result["normalized_prediction"], "A, C")
        over = score_answer(sample, '{"correct_options":["A","B","C"]}')
        under = score_answer(sample, '{"correct_options":["A"]}')
        self.assertEqual(over["answer_accuracy"], 0.0)
        self.assertAlmostEqual(over["choice_f1"], 0.8)
        self.assertEqual(under["answer_accuracy"], 0.0)
        self.assertAlmostEqual(under["choice_f1"], 2 / 3)

    def test_multiple_choice_does_not_accept_disjunction_or_pick_letters_from_explanation(self):
        for prediction in ("A or C", "not A and C", "A is incorrect; C is correct", "A and Z"):
            with self.subTest(prediction=prediction):
                result = score_answer(choice_sample(multiple=True), prediction)
                self.assertEqual(result["parse_status"], "prediction_unparseable")
                self.assertEqual(result["answer_accuracy"], 0.0)
                self.assertEqual(result["choice_f1"], 0.0)

    def test_real_mira_multiple_choice_overselection_has_zero_exact_accuracy(self):
        options = ["A. There is significant apical ballooning.", "B. The basal segments are hypercontractile.",
                   "C. There is evidence of coronary artery disease.",
                   "D. The findings are consistent with acute myocardial infarction."]
        sample = choice_sample(multiple=True, options=options, reference={"correct_options": ["A", "B"]})
        prediction = json.dumps({"correct_options": options[:3], "explanation": "The angiogram shows infarction.",
                                 "visual_evidence": "thrombus"})
        result = score_answer(sample, prediction)
        self.assertEqual(result["answer_accuracy"], 0.0)
        self.assertAlmostEqual(result["choice_f1"], 0.8)
        self.assertEqual(result["normalized_prediction"], "A, B, C")

    def test_real_yes_no_ignores_visual_evidence_and_scores_actual_answer(self):
        sample = {"question_type": "closed_ended", "answer": {"text": "Yes", "visual_evidence": "wide QRS"}}
        prediction = '{"text":"No","visual_evidence":"There is right bundle branch block."}'
        result = score_answer(sample, prediction)
        self.assertEqual(result["answer_accuracy"], 0.0)
        self.assertEqual(result["normalized_prediction"], "No")
        self.assertEqual(result["normalized_reference"], "Yes")
        for answer in ("Yes", "Yes, RBBB is visible.", "答案：是，存在右束支传导阻滞。", "Final answer: Yes"):
            with self.subTest(answer=answer):
                self.assertEqual(score_answer(sample, answer)["answer_accuracy"], 1.0)

    def test_yes_no_does_not_guess_from_semantic_description_or_uncertainty(self):
        sample = {"question_type": "closed_ended", "answer": {"text": "Yes"}}
        for prediction in ("Not yes", "Maybe yes", "Yes or no", "Yes, actually no.", "Yes/no", "The ECG shows RBBB."):
            with self.subTest(prediction=prediction):
                result = score_answer(sample, prediction)
                self.assertEqual(result["parse_status"], "prediction_unparseable")
                self.assertEqual(result["answer_accuracy"], 0.0)
        self.assertEqual(score_answer({"question_type": "closed_ended", "answer": False}, "No")["answer_accuracy"], 1.0)

    def test_open_text_metrics_use_text_field_without_json_keys_or_visual_evidence(self):
        sample = {"question_type": "open_ended", "answer": {"text": "The imaging modality is coronary angiography.",
                                                              "visual_evidence": "arrow evidence"}}
        prediction = '{"text":"Coronary angiography.","visual_evidence":"different evidence"}'
        result = score_answer(sample, prediction)
        self.assertEqual(result["normalized_prediction"], "Coronary angiography.")
        self.assertEqual(result["normalized_reference"], "The imaging modality is coronary angiography.")
        self.assertFalse(result["eligible"])
        self.assertIsNone(result["answer_accuracy"])
        self.assertEqual(result["parse_status"], "not_objective")

    def test_ambiguous_reference_is_not_eligible_and_does_not_lower_denominator(self):
        samples = [choice_sample(reference={"correct_option": "C or B"}),
                   choice_sample(reference={"explanation": "It might be C"}),
                   choice_sample(reference={"correct_option": "Z"}),
                   {"question_type": "closed_ended", "answer": {"text": "Possibly yes"}},
                   {"question_type": "closed_ended", "answer": {"text": "Yes or No"}}]
        for sample in samples:
            with self.subTest(sample=sample):
                result = score_answer(sample, "C")
                self.assertFalse(result["eligible"])
                self.assertIsNone(result["answer_accuracy"])
                self.assertEqual(result["parse_status"], "invalid_reference")

    def test_complete_think_is_removed_but_unclosed_or_think_only_is_invalid(self):
        for prediction in ("<think>Maybe A or B. </think>C", "<think>Not C.</think>\nFinal answer: C"):
            with self.subTest(prediction=prediction):
                self.assertEqual(score_answer(choice_sample(), prediction)["answer_accuracy"], 1.0)
        for prediction in ("<think>C", "<think>C</think>", "C<think>unfinished", "C</think>"):
            with self.subTest(prediction=prediction):
                result = score_answer(choice_sample(), prediction)
                self.assertEqual(result["answer_accuracy"], 0.0)
                self.assertEqual(result["normalized_prediction"], "")
                self.assertEqual(result["parse_status"], "prediction_unparseable")

    def test_legacy_saved_records_infer_type_parse_reference_and_recover_options(self):
        sample = {"id": "mira:train:5463:single_choice:0",
                  "question": 'What modality?\nOptions: ["A. MRI","B. CT","C. Echocardiography","D. X-ray"]',
                  "reference": '{"correct_option":"C. Echocardiography","explanation":"Doppler"}'}
        result = score_answer(sample, "C. Echocardiography")
        self.assertEqual(result["question_type"], "single_choice")
        self.assertEqual(result["answer_accuracy"], 1.0)

    def test_dict_and_unlabelled_options_supported_but_ambiguous_text_is_not_guessed(self):
        sample = choice_sample(options={"A": "MRI", "B": "CT", "C": "Echocardiography"})
        self.assertEqual(score_answer(sample, "C")["answer_accuracy"], 1.0)
        sample = choice_sample(options=["MRI", "CT", "Echocardiography"])
        self.assertEqual(score_answer(sample, "Echocardiography")["answer_accuracy"], 1.0)
        sample = choice_sample(options={"A": "same", "B": "same"}, reference={"correct_option": "same"})
        self.assertFalse(score_answer(sample, "A")["eligible"])

    def test_real_closed_references_with_explanations_are_all_eligible(self):
        references = [
            ("Yes, the imaging modality used here, which is a delayed portal venogram during TIPS creation, "
             "is highly suitable for evaluating portal hypertension.", "Yes"),
            ("No, the ventriculogram does not show signs of acute myocardial infarction. "
             "The pattern of apical ballooning is more consistent with Takotsubo cardiomyopathy.", "No"),
            ("Yes, the OCT-Angiography shows signs of back-shadowing at the level of the choriocapillaris.", "Yes"),
        ]
        for reference, answer in references:
            with self.subTest(reference=reference):
                result = score_answer({"question_type": "closed_ended", "answer": {"text": reference}}, answer)
                self.assertTrue(result["eligible"])
                self.assertEqual(result["answer_accuracy"], 1)
                self.assertEqual(result["normalized_reference"], answer)
                self.assertEqual(result["semantic_reference"], reference)

    def test_real_multiple_choice_can_use_singular_correct_option(self):
        options = ["A. Immediate need for anticoagulation therapy", "B. Risk of right heart failure",
                   "C. Potential for recurrent emboli", "D. All of the above"]
        sample = choice_sample(multiple=True, options=options, reference={"correct_option": "D. All of the above"})
        for prediction in ("D. All of the above", "Therefore, the correct answer is:\nD. All of the above"):
            with self.subTest(prediction=prediction):
                result = score_answer(sample, prediction)
                self.assertTrue(result["eligible"])
                self.assertEqual(result["answer_accuracy"], 1)
                self.assertEqual(result["normalized_reference"], "D")
        self.assertEqual(score_answer(sample, "A")["answer_accuracy"], 0)

    def test_real_independent_answer_paragraph_is_parsed_even_when_wrong(self):
        sample = choice_sample(options=["A. Hypercoagulable state", "B. Trauma", "C. Atherosclerosis", "D. Infection"],
                               reference={"correct_option": "A. Hypercoagulable state"})
        prediction = ("Based on the image provided, the most likely cause of the observed thrombus is:\n\n"
                      "**B. Trauma**\n\n**Rationale:**\nThe image shows a thrombus in a vessel.\n\n"
                      "Therefore, the most likely cause of the observed thrombus is **B. Trauma**.")
        result = score_answer(sample, prediction)
        self.assertEqual(result["parse_status"], "ok")
        self.assertEqual(result["normalized_prediction"], "B")
        self.assertEqual(result["answer_accuracy"], 0, "解析不能根据参考答案改成A。")

    def test_real_is_colon_and_newline_answer_markers(self):
        sample = choice_sample(options=["A. Left main coronary artery", "B. Right coronary artery"],
                               reference={"correct_option": "B"})
        predictions = [
            "The image is discussed.\n\nTherefore, the correct answer is:\nA. Left main coronary artery",
            "Based on the provided image, the correct answer is:\n\n**A. Left main coronary artery**\n\nExplanation: details.",
            "Answer:\nA. Left main coronary artery\nExplanation: details.",
        ]
        for prediction in predictions:
            with self.subTest(prediction=prediction):
                result = score_answer(sample, prediction)
                self.assertEqual(result["parse_status"], "ok")
                self.assertEqual(result["normalized_prediction"], "A")
                self.assertEqual(result["answer_accuracy"], 0)

    def test_real_natural_choice_assertions_and_bare_option_labels(self):
        sample = choice_sample(options=["A. Stroke", "B. Hemorrhage"], reference={"correct_option": "B"})
        result = score_answer(sample, "The primary risk associated with the observed pathology is A. Stroke.\n\nClinical explanation.")
        self.assertEqual(result["parse_status"], "ok")
        self.assertEqual(result["normalized_prediction"], "A")
        self.assertEqual(result["answer_accuracy"], 0)
        sample = choice_sample(options=["A", "B", "C", "D"], reference={"correct_option": "A"})
        for prediction in ("Based on the image, the most commonly used initial treatment is **A**.\n\nDescription.",
                           "The most likely cause is:\n- **A**. The correct answer is **A**.\n\nExplanation."):
            with self.subTest(prediction=prediction):
                self.assertEqual(score_answer(sample, prediction)["answer_accuracy"], 1)

    def test_negative_candidate_lists_and_conflicting_final_answers_stay_unparseable(self):
        sample = choice_sample()
        predictions = [
            "notC", "Answer: C\n\nC is not correct.", "Answer: C\n\nActually, B. CT",
            "A. MRI\n\nFinal answer: C", "Answer: A\nFinal answer: C",
            "My diagnosis is:\n\nA. MRI\n\nFinal answer: C",
            "The options are:\nA. MRI\nB. CT\nC. Echocardiography\nD. X-ray",
            "Candidates:\n\nC. Echocardiography", "The wrong option is C. Echocardiography",
        ]
        for prediction in predictions:
            with self.subTest(prediction=prediction):
                result = score_answer(sample, prediction)
                self.assertEqual(result["parse_status"], "prediction_unparseable")
                self.assertEqual(result["answer_accuracy"], 0)
        closed = {"question_type": "closed_ended", "answer": "Yes"}
        for prediction in ("Yes.\n\nNo.", "Yes, it is visible.\n\nFinal answer: No", "Answer: Yes\nAnswer: No"):
            with self.subTest(prediction=prediction):
                self.assertEqual(score_answer(closed, prediction)["parse_status"], "prediction_unparseable")

    def test_semantic_fields_preserve_explanations_without_thinking_or_image_annotations(self):
        sample = choice_sample(reference={"correct_option": "C", "explanation": "Shows the chamber and valve motion.",
                                          "visual_evidence": "REFERENCE_IMAGE_ONLY"})
        prediction = '<think>SECRET_REASONING</think>' + json.dumps({
            "correct_option": "C", "text": "Echocardiography.",
            "explanation": "This is CT; an important clinical contradiction.", "visual_evidence": "PREDICTION_IMAGE_ONLY",
        })
        result = score_answer(sample, prediction)
        self.assertEqual(result["normalized_prediction"], "C")
        self.assertEqual(result["answer_accuracy"], 1)
        self.assertIn("Echocardiography.", result["semantic_prediction"])
        self.assertIn("clinical contradiction", result["semantic_prediction"])
        self.assertIn("chamber and valve", result["semantic_reference"])
        self.assertNotIn("SECRET_REASONING", result["semantic_prediction"])
        self.assertNotIn("IMAGE_ONLY", result["semantic_prediction"] + result["semantic_reference"])
        self.assertEqual(score_answer(sample, "<think>unfinished")["semantic_prediction"], "")

    def test_explicit_answer_label_allows_same_line_explanation(self):
        sample = choice_sample()
        predictions = [
            "Answer: C because it assesses valve motion.",
            "Answer: C, because it assesses valve motion.",
            "Answer: C. Echocardiography is the correct modality.",
            "Answer: C. Echocardiography provides real-time valve assessment.",
            "Answer: C. This is supported by the moving valves.",
            "Final answer: C. Explanation: the chambers are shown in motion.",
            "答案：C，因为可评估瓣膜运动。",
        ]
        for prediction in predictions:
            with self.subTest(prediction=prediction):
                result = score_answer(sample, prediction)
                self.assertEqual(result["parse_status"], "ok")
                self.assertEqual(result["answer_accuracy"], 1)
                self.assertEqual(result["normalized_prediction"], "C")
                self.assertEqual(result["semantic_prediction"], prediction)
        wrong = score_answer(sample, "Answer: B because CT provides cross-sectional images.")
        self.assertEqual(wrong["parse_status"], "ok")
        self.assertEqual(wrong["answer_accuracy"], 0)
        self.assertEqual(wrong["normalized_prediction"], "B")

    def test_same_line_explanation_does_not_hide_uncertain_or_conflicting_choice(self):
        predictions = [
            "Answer: C or B because either may fit.",
            "Answer: maybe C because it shows motion.",
            "Answer: not C because CT is used.",
            "Answer: C. CT", "Answer: C. Not echocardiography.",
            "Answer: C. Echocardiography is not correct.",
            "Answer: C because B is actually the correct answer.",
            "Answer: C because actually B.",
            "Answer: C because the answer is B.",
            "Answer: C because perhaps it is an echocardiogram.",
            "Answer: C. This is supported.\nFinal answer: B",
            "C because it assesses valve motion.",
            "During reasoning I considered C because it assesses valve motion.",
        ]
        for prediction in predictions:
            with self.subTest(prediction=prediction):
                result = score_answer(choice_sample(), prediction)
                self.assertEqual(result["parse_status"], "prediction_unparseable")
                self.assertEqual(result["answer_accuracy"], 0)

    def test_explicit_multiple_choice_first_line_allows_next_line_or_because_explanation(self):
        sample = choice_sample(multiple=True)
        for prediction in (
            "Answer: A, C\nBoth findings are supported.",
            "Answer: C and A\nMRI and echo assess the relevant structures.",
            "Answer: A, C because both findings are supported.",
            "Answer: A, C. because both findings are supported.",
            "答案：A、C，因为两项均有依据。",
            "Answer: A, C\nBoth are supported.\nFinal answer: C, A",
            "Answer: A\nC",
        ):
            with self.subTest(prediction=prediction):
                result = score_answer(sample, prediction)
                self.assertEqual(result["parse_status"], "ok")
                self.assertEqual(result["answer_accuracy"], 1)
                self.assertEqual(result["choice_f1"], 1)
                self.assertEqual(result["normalized_prediction"], "A, C")
                self.assertEqual(result["semantic_prediction"], prediction)
        wrong = score_answer(sample, "Answer: A, B\nBoth findings are supported.")
        self.assertEqual(wrong["parse_status"], "ok")
        self.assertEqual(wrong["answer_accuracy"], 0)
        self.assertEqual(wrong["choice_f1"], 0.5)

    def test_multiple_choice_first_line_never_discards_extra_choices_or_conflicts(self):
        predictions = [
            "Answer: A, C\nB", "Answer: A, C\nBoth are supported.\nB. CT",
            "Answer: A, C\nBoth are supported.\nFinal answer: A, B",
            "Answer: A, C\nBoth are supported.\n\nA, B",
            "Answer: A, C\nZ", "Answer: A, C because B is the correct answer.",
            "Answer: A, C because perhaps either might fit.",
            "Answer: A or C\nBoth are discussed.", "Answer: not A, C\nBoth are discussed.",
            "A, C\nBoth findings are supported.", "I considered A, C because both fit.",
        ]
        for prediction in predictions:
            with self.subTest(prediction=prediction):
                result = score_answer(choice_sample(multiple=True), prediction)
                self.assertEqual(result["parse_status"], "prediction_unparseable")
                self.assertEqual(result["answer_accuracy"], 0)


if __name__ == "__main__":
    unittest.main()
