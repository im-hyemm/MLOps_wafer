from sklearn.metrics import classification_report
from config.settings import CLASSES, ID2LABEL

def calculate_metrics(y_true, y_pred):
    """분류 성능 지표 계산"""
    labels = sorted(ID2LABEL.keys())
    target_names = [ID2LABEL[i] for i in labels]
    
    report = classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=target_names, 
        zero_division=0,
        output_dict=True,
    )

    metrics_by_cat = {}
    for cls_id in labels:
        name = ID2LABEL[cls_id]
        cls_block = report.get(name, None)

        if cls_block is None:
            metrics_by_cat[str(cls_id)] = {
                "label": name,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "support": 0,
            }
        else:
            metrics_by_cat[str(cls_id)] = {
                "label": name,
                "precision": round(float(cls_block.get("precision", 0.0)), 2),
                "recall":    round(float(cls_block.get("recall", 0.0)), 2),
                "f1":        round(float(cls_block.get("f1-score", 0.0)), 2),
                "support":   int(cls_block.get("support", 0)),
            }    
            
    test_acc = round(float(report.get("accuracy", 0.0)), 2)
    test_macro_precision = round(float(report["macro avg"].get("precision", 0.0)), 2)
    test_macro_recall = round(float(report["macro avg"].get("recall", 0.0)), 2)
    test_macro_f1 = round(float(report["macro avg"].get("f1-score", 0.0)), 2)

    return test_acc, test_macro_f1, test_macro_precision, test_macro_recall, metrics_by_cat