import random
import pandas as pd 
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datetime import datetime
from sklearn.metrics import mean_absolute_error

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

    print("Starting Evaluation.....")
    print("Submission related metadata:")
    print(kwargs['submission_metadata'])

    #print(test_annotation_file)
    #print(user_submission_file)
    up = pd.read_csv(user_submission_file,sep=',')
    gt = pd.read_csv(test_annotation_file,sep=',')
    
    gt.columns = [col.lower() for col in gt.columns]
    up.columns = [col.lower() for col in up.columns]

    merged = pd.merge(gt, up, how='left', on=['date','hostname','type'])
    print(merged.shape[0] , gt.shape[0])

    if up.shape[0] != gt.shape[0] :
        print(up.shape[0] , gt.shape[0])
        print ("PROBLEME : identifiant different du fichier attendu, le nombre de ligne du fichier soumis ne semble pas bon ou les ids ne matche pas, verifier avec le fichier de test pour la bonne syntaxe")
        exit


    merged.fillna(0,inplace=True)

    y_true = merged['inactive_x'].values
    y_pred = merged['inactive_y'].values

    print('calculating metrics...')
    
    F1 =  f1_score(y_true, y_pred)
    
    print('F1:%8.5f' % (F1))

   
    output = {}
    if 1:
        print("Evaluating for Test Phase")
        output['result'] = [
            {
                'test': { 'score': F1
                         }
            }
        ]
        print("Completed evaluation for Test Phase")
    output['submission_metadata'] = "This submission metadata will only be shown to the Challenge Host"
    output['submission_result'] = "This is the actual result to show to the participant once submission is finished"
    return output

