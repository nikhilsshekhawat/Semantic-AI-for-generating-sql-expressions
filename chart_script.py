import plotly.graph_objects as go
import plotly.express as px
import json

# Parse the provided data
data = {
    "architecture_components": [
        {"layer": "Input Processing", "components": ["Natural Language Query", "Database Schema", "Context Information"], "color": "#3498db"}, 
        {"layer": "AI Model Layer", "components": ["GPT-4/GPT-3.5", "T5/T5-Large", "SQLformer", "Custom Fine-tuned Models", "RAG Components"], "color": "#2ecc71"}, 
        {"layer": "Processing Pipeline", "components": ["Schema Linking", "Query Generation", "SQL Validation", "Error Correction"], "color": "#e67e22"}, 
        {"layer": "Output Layer", "components": ["SQL Query", "Execution Results", "Explanations"], "color": "#9b59b6"}, 
        {"layer": "Training Components", "components": ["Training Datasets", "Fine-tuning", "Evaluation"], "color": "#e74c3c"}
    ]
}

# Create data for the scatter plot
x_positions = []
y_positions = []
labels = []
colors = []
hover_texts = []
sizes = []

# Define y-positions for each layer (top to bottom flow)
layer_y_positions = {
    "Input Processing": 4,
    "AI Model Layer": 3,
    "Processing Pipeline": 2,
    "Output Layer": 1,
    "Training Components": 0
}

# Process each layer
for layer_data in data["architecture_components"]:
    layer_name = layer_data["layer"]
    components = layer_data["components"]
    layer_color = layer_data["color"]
    y_pos = layer_y_positions[layer_name]
    
    # Calculate x positions with better spacing
    num_components = len(components)
    if num_components == 1:
        x_positions.append(0)
    else:
        # Use wider spacing for better readability
        spacing = 1.5 if num_components <= 3 else 1.2
        x_start = -(num_components - 1) * spacing / 2
        for i in range(num_components):
            x_positions.append(x_start + i * spacing)
    
    # Add components
    for component in components:
        y_positions.append(y_pos)
        # Truncate labels to 15 characters but keep them readable
        short_label = component[:15]
        labels.append(short_label)
        colors.append(layer_color)
        hover_texts.append(f"{layer_name}<br>{component}")
        sizes.append(60)  # Increased size for better readability

# Create the scatter plot
fig = go.Figure()

# Add scatter points for each component
fig.add_trace(go.Scatter(
    x=x_positions,
    y=y_positions,
    mode='markers+text',
    text=labels,
    textposition='middle center',
    marker=dict(
        size=sizes,
        color=colors,
        line=dict(width=3, color='white'),
        symbol='square'
    ),
    hovertemplate='%{hovertext}<extra></extra>',
    hovertext=hover_texts,
    textfont=dict(size=11, color='white', family='Arial Black'),  # Increased font size and weight
    name='',
    showlegend=False
))

# Add layer labels on the left side
layer_names = ["Input Processing", "AI Model Layer", "Processing Pipeline", "Output Layer", "Training Components"]
layer_positions = [4, 3, 2, 1, 0]

for i, (name, pos) in enumerate(zip(layer_names, layer_positions)):
    fig.add_trace(go.Scatter(
        x=[-4],
        y=[pos],
        mode='text',
        text=[name[:15]],  # Truncate to 15 chars
        textposition='middle right',
        textfont=dict(size=13, color='black', family='Arial'),
        showlegend=False,
        hoverinfo='none'
    ))

# Add connecting arrows between layers
arrow_positions = [3.5, 2.5, 1.5, 0.5]
for y_pos in arrow_positions:
    fig.add_trace(go.Scatter(
        x=[0, 0],
        y=[y_pos + 0.3, y_pos - 0.3],
        mode='lines',
        line=dict(color='gray', width=4),
        showlegend=False,
        hoverinfo='none'
    ))
    
    # Add arrowhead
    fig.add_trace(go.Scatter(
        x=[0],
        y=[y_pos - 0.3],
        mode='markers',
        marker=dict(
            symbol='triangle-down',
            size=15,
            color='gray'
        ),
        showlegend=False,
        hoverinfo='none'
    ))

# Update layout
fig.update_layout(
    title='AI Text-to-SQL Architecture',
    xaxis=dict(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        range=[-5, 5]  # Expanded range for better layout
    ),
    yaxis=dict(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        range=[-0.5, 4.5]
    ),
    plot_bgcolor='rgba(0,0,0,0)'
)

# Update trace settings
fig.update_traces(cliponaxis=False)

# Save the chart
fig.write_image('text_to_sql_architecture.png')
