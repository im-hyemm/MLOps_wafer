import plotly.express as px
from sklearn.metrics import confusion_matrix
from config.settings import CLASSES

def draw_cm_heatmap(true_labels, pred_labels, save_path, palette='Peach', normalize='true'):
    """Confusion Matrix 히트맵 생성"""
    cm = confusion_matrix(true_labels, pred_labels, normalize=normalize)
    
    fig = px.imshow(cm,
                    labels=dict(
                        x="Prediction Label",
                        y="True Label",
                        color="Normalized Count"
                    ),
                    x=CLASSES,
                    y=CLASSES,
                    color_continuous_scale=palette,
                    text_auto='.2f'
                    )
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        font=dict(size=12),
        xaxis=dict(tickangle=45),
    )
    
    fig.write_html(save_path)
    print(f"Plotly confusion matrix saved to {save_path}")

    return fig