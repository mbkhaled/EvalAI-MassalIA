import pandas as pd 
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

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

    test_data = pd.read_csv(test_annotation_file)
    user_data = pd.read_csv(user_submission_file)
    #TODO : ajouter contrôles fichier et son contenu

    score = mean_squared_error(test_data.Voltage  ,user_data.Voltage)
    r2 = r2_score(test_data.Voltage  ,user_data.Voltage)

    output = {}
    print("Evaluating for Dev Phase")

    output["result"] = [
        {
            "train_split": {
                "MSE": score
                "R2": r2
            }
        }
    ]
    # To display the results in the result file
    print("le MSE pour cette soumission est : ",score)
    print("le R2  est : ",r2)
    output["submission_result"] = output["result"][0]["train_split"]
    print("Completed evaluation")

    return output

