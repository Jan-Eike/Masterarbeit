from sklearn.metrics import f1_score
import numpy as np

def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_metrics(p):    
    preds, labels = p
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {"f1": f1} 


def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities, also not sure about just rounding...
    return 'f1', f1_score(y_true, y_hat, average='macro'), True