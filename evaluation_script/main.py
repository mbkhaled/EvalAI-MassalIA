import pandas as pd 
import numpy as np
from sklearn.metrics import mean_squared_error

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

    # dev phase scores 
    score_active_power = mean_squared_error(test_data.Global_active_power  ,user_data.Global_active_power)
    score_reactive_power = mean_squared_error(test_data.Global_reactive_power  ,user_data.Global_reactive_power)
    score_voltage = mean_squared_error(test_data.Voltage  ,user_data.Voltage)
    score_intensity = mean_squared_error(test_data.Global_intensity  ,user_data.Global_intensity)
    score_sub_metering_1 = mean_squared_error(test_data.Sub_metering_1  ,user_data.Sub_metering_1)
    score_sub_metering_2 = mean_squared_error(test_data.Sub_metering_2  ,user_data.Sub_metering_3)
    score_sub_metering_3 = mean_squared_error(test_data.Sub_metering_2  ,user_data.Sub_metering_3)
    score_overall = np.mean([
        score_active_power,
        score_reactive_power,
        score_voltage,
        score_intensity,
        score_sub_metering_1,
        score_sub_metering_2,
        score_sub_metering_3
    ])



    output = {}
    print("Evaluating for Dev Phase")

    output["result"] = [
        {
            "train_split": {
                "Active Power MSE": score_active_power,
                "Reactive Power MSE": score_reactive_power,
                "Voltage MSE": score_voltage,
                "Global Intensity MSE": score_intensity,
                "Sub_metering_1 MSE": score_sub_metering_1,
                "Sub_metering_2 MSE": score_sub_metering_2,
                "Sub_metering_3 MSE": score_sub_metering_3,
                "Overall MSE": score_overall,
            }
        }
    ]
    # To display the results in the result file
    output["submission_result"] = output["result"][0]["train_split"]
    print("Completed evaluation")

    return output

