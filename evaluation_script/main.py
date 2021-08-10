import csv
import numpy as np
from mean_average_precision import MetricBuilder

def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    """
    Evaluates the submission for a particular challenge phase adn returns score
    Arguments:

        `test_annotations_file`: Path to test_annotation_file on the server
        `user_submission_file`: Path to file submitted by the user
        `phase_codename`: Phase to which submission is made

        `**kwargs`: keyword arguments that contains additional submission
        metadata that challenge hosts can use to send slack notification.
        You can access the submission metadata
        with kwargs['submission_metadata']

        Example: A sample submission metadata can be accessed like this:
        >>> print(kwargs['submission_metadata'])
        {
            'status': u'running',
            'when_made_public': None,
            'participant_team': 5,
            'input_file': 'https://abc.xyz/path/to/submission/file.json',
            execution_time': u'123',
            'publication_url': u'ABC',
            'challenge_phase': 1,
            'created_by': u'ABC',
            'stdout_file': 'https://abc.xyz/path/to/stdout/file.json',
            'method_name': u'Test',
            'stderr_file': 'https://abc.xyz/path/to/stderr/file.json',
            'participant_team_name': u'Test Team',
            'project_url': u'http://foo.bar',
            'method_description': u'ABC',
            'is_public': False,
            'submission_result_file': 'https://abc.xyz/path/result/file.json',
            'id': 123,
            'submitted_at': u'2017-03-20T19:22:03.880652Z'
        }
    """
    pred_boxes = dict()
    with open(user_submission_file, newline='') as csvfile:
        predictions = csv.reader(csvfile, delimiter=',')
        next(predictions)
        for row in predictions:
            l = pred_boxes.get(row[0], list())
            l.append(row[1:])
            pred_boxes[row[0]] = l

    true_boxes = dict()
    with open(test_annotation_file, newline='') as csvfile:
        test = csv.reader(csvfile, delimiter=',')
        next(test)
        for row in test:
            l = true_boxes.get(row[0], list())
            l.append(row[1:]+[0,0,0])
            true_boxes[row[0]] = l

    metric_fn = MetricBuilder.build_evaluation_metric('map_2d', async_mode=True, num_classes=1)
    for img in pred_boxes:
        metric_fn.add(np.array(pred_boxes[img]).astype(float), np.array(true_boxes[img]).astype(float))
    score = metric_fn.value(iou_thresholds=0.5)['mAP']
    score = int(score*100)
    
    output = {}
    print("Evaluating for Antenna Detection Phase")

    output["result"] = [
        {
            "train_antenna": {
                "mAP": score,
            }
        }
    ]
    # To display the results in the result file
    print("la mAP pour cette soumission est : ",score)
    output["submission_result"] = output["result"][0]
    print("Completed evaluation")

    return output

