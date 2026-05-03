import pandas as pd

# Load entities
df = pd.read_csv("disaster_entities.csv")

# Filter locations (GPE)
gpe_df = df[df['entity_label'] == 'GPE']

# Count frequency
location_counts = gpe_df['entity'].value_counts().reset_index()
location_counts.columns = ['Location', 'Frequency']

# Save results
location_counts.to_csv("location_analysis.csv", index=False)

print("Location analysis completed!")
print("Saved as location_analysis.csv")