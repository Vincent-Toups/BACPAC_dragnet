import pandas as pd
from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, HoverTool, FactorRange
from bokeh.layouts import gridplot

# Load Data
projection = pd.read_csv("derived_data/demo_vae_projection.csv")
subject_changes = pd.read_csv("derived_data/subject-changes.csv")

# Merge and Process Data
projection_ex = projection.merge(subject_changes, on="USUBJID", how="inner")
projection_ex.to_csv("derived_data/demographics-with-projection.csv", index=False)

# Plot 2D Projection by Race and Ethnicity
source = ColumnDataSource(projection)

p1 = figure(title="Demographic 2D Projection",
            x_axis_label="E1", y_axis_label="E2",
            tools="pan,zoom_in,zoom_out,reset,save", width=700, height=500)
p1.circle(x="E1", y="E2", source=source, size=10, alpha=0.6,
          color="blue", legend_field="RACE, ETHNICITY")

p1.legend.title = "Race & Ethnicity"
p1.add_tools(HoverTool(tooltips=[("E1", "@E1"), ("E2", "@E2"), ("Race, Ethnicity", "@{RACE, ETHNICITY}")]))

# Plot by Study ID
p2 = figure(title="2D Projection by Study ID",
            x_axis_label="E1", y_axis_label="E2",
            tools="pan,zoom_in,zoom_out,reset,save", width=700, height=500)
p2.circle(x="E1", y="E2", source=source, size=10, alpha=0.6,
          color="green", legend_field="STUDYID (Count)")

p2.legend.title = "Study ID"
p2.add_tools(HoverTool(tooltips=[("E1", "@E1"), ("E2", "@E2"), ("Study ID", "@{STUDYID (Count)}")]))

# Faceted Plot by Study ID
unique_study_ids = projection["STUDYID (Count)"].unique()
plots = []
for study_id in unique_study_ids:
    sub_df = projection[projection["STUDYID (Count)"] == study_id]
    source_sub = ColumnDataSource(sub_df)
    
    p = figure(title=f"Study ID: {study_id}",
               x_axis_label="E1", y_axis_label="E2",
               tools="pan,zoom_in,zoom_out,reset,save", width=350, height=350)
    p.circle(x="E1", y="E2", source=source_sub, size=10, alpha=0.6, color="purple")
    p.add_tools(HoverTool(tooltips=[("E1", "@E1"), ("E2", "@E2"), ("Count", f"{study_id}")]))
    plots.append(p)

# Create a grid layout for faceted plots
grid = gridplot([plots[i:i+2] for i in range(0, len(plots), 2)], sizing_mode="scale_width")

# Save Outputs
output_file("figures/bokeh_projection_plots.html")
