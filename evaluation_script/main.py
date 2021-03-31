import pandas as pd 
import numpy as np
from sklearn.metrics import mean_squared_error

def mean_absolute_percentage_error(y_true, y_pred,
                                   sample_weight=None,
                                   multioutput='uniform_average'):
    """Mean absolute percentage error regression loss.
    Note here that we do not represent the output as a percentage in range
    [0, 100]. Instead, we represent it in range [0, 1/eps]. Read more in the
    :ref:`User Guide <mean_absolute_percentage_error>`.
    .. versionadded:: 0.24
    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    multioutput : {'raw_values', 'uniform_average'} or array-like
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
        If input is list then the shape must be (n_outputs,).
        'raw_values' :
            Returns a full set of errors in case of multioutput input.
        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.
    Returns
    -------
    loss : float or ndarray of floats in the range [0, 1/eps]
        If multioutput is 'raw_values', then mean absolute percentage error
        is returned for each output separately.
        If multioutput is 'uniform_average' or an ndarray of weights, then the
        weighted average of all output errors is returned.
        MAPE output is non-negative floating point. The best value is 0.0.
        But note the fact that bad predictions can lead to arbitarily large
        MAPE values, especially if some y_true values are very close to zero.
        Note that we return a large value instead of `inf` when y_true is zero.
    Examples
    --------
    >>> from sklearn.metrics import mean_absolute_percentage_error
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> mean_absolute_percentage_error(y_true, y_pred)
    0.3273...
    >>> y_true = [[0.5, 1], [-1, 1], [7, -6]]
    >>> y_pred = [[0, 2], [-1, 2], [8, -5]]
    >>> mean_absolute_percentage_error(y_true, y_pred)
    0.5515...
    >>> mean_absolute_percentage_error(y_true, y_pred, multioutput=[0.3, 0.7])
    0.6198...
    """
    #y_type, y_true, y_pred, multioutput = _check_reg_targets(  y_true, y_pred, multioutput)
    #check_consistent_length(y_true, y_pred, sample_weight)
    epsilon = np.finfo(np.float64).eps
    mape = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), epsilon)
    output_errors = np.average(mape,
                               weights=sample_weight, axis=0)
    if isinstance(multioutput, str):
        if multioutput == 'raw_values':
            return output_errors
        elif multioutput == 'uniform_average':
            # pass None as weights to np.average: uniform mean
            multioutput = None

    return np.average(output_errors, weights=multioutput)





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
    user_data = pd.read_csv(user_annotation_file)
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

