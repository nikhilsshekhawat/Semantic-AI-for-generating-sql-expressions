import plotly.graph_objects as go
import json

# Data from the JSON
data = {
    "training_phases": [
        {"phase": "Data Collection", "components": ["WikiSQL Dataset (87K)", "Spider Dataset (10K)", "Synthetic Data", "Domain Data"], "color": "#3498db"},
        {"phase": "Preprocessing", "components": ["Schema Linking", "Query Normalization", "Data Cleaning", "Train/Val/Test Split"], "color": "#e67e22"},
        {"phase": "Model Training", "components": ["Base Model Selection", "Fine-tuning Process", "Hyperparameter Tuning", "Validation"], "color": "#2ecc71"},
        {"phase": "Evaluation", "components": ["Execution Accuracy", "Exact Match", "Schema Linking", "Performance Metrics"], "color": "#9b59b6"},
        {"phase": "Deployment", "components": ["Model Optimization", "Inference Pipeline", "Production Deploy"], "color": "#e74c3c"}
    ]
}

# Use the brand colors
brand_colors = ["#1FB8CD", "#FFC185", "#ECEBD5", "#5D878F", "#D2BA4C"]

# Create the chart
fig = go.Figure()

# Add rectangles and text for each phase
for i, phase_data in enumerate(data["training_phases"]):
    phase_name = phase_data["phase"]
    components = phase_data["components"]
    
    # Truncate phase name to 15 characters
    if len(phase_name) > 15:
        phase_name = phase_name[:12] + "..."
    
    y_pos = len(data["training_phases"]) - i
    
    # Add rectangle shape for each phase
    fig.add_shape(
        type="rect",
        x0=0.1, y0=y_pos - 0.45,
        x1=1.9, y1=y_pos + 0.45,
        fillcolor=brand_colors[i],
        line=dict(color="black", width=2),
        opacity=0.9
    )
    
    # Truncate components to 15 chars each
    truncated_components = []
    for comp in components:
        if len(comp) > 15:
            truncated_components.append(comp[:12] + "...")
        else:
            truncated_components.append(comp)
    
    # Add phase title
    fig.add_trace(go.Scatter(
        x=[1],
        y=[y_pos + 0.25],
        mode='text',
        text=phase_name,
        textfont=dict(size=14, color="black", family="Arial Black"),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Add components as separate text traces
    for j, comp in enumerate(truncated_components):
        fig.add_trace(go.Scatter(
            x=[1],
            y=[y_pos + 0.05 - (j * 0.15)],
            mode='text',
            text=f"• {comp}",
            textfont=dict(size=10, color="black"),
            showlegend=False,
            hoverinfo='skip'
        ))

# Add arrows between phases
for i in range(len(data["training_phases"]) - 1):
    y_start = len(data["training_phases"]) - i - 0.45
    y_end = len(data["training_phases"]) - i - 0.55
    
    # Add arrow line
    fig.add_shape(
        type="line",
        x0=1, y0=y_start,
        x1=1, y1=y_end,
        line=dict(
            color="black",
            width=4
        )
    )
    
    # Add arrowhead
    fig.add_shape(
        type="path",
        path=f"M 0.85,{y_end + 0.08} L 1.15,{y_end + 0.08} L 1,{y_end} Z",
        fillcolor="black",
        line=dict(color="black", width=0)
    )

# Update layout
fig.update_layout(
    title="AI Text-to-SQL Training Flow",
    xaxis=dict(
        showgrid=False,
        showticklabels=False,
        zeroline=False,
        range=[0, 2],
        fixedrange=True
    ),
    yaxis=dict(
        showgrid=False,
        showticklabels=False,
        zeroline=False,
        range=[0.3, len(data["training_phases"]) + 0.7],
        fixedrange=True
    ),
    showlegend=False,
    plot_bgcolor='white',
    paper_bgcolor='white'
)

# Save the chart
fig.write_image("ai_text_to_sql_flowchart.png")