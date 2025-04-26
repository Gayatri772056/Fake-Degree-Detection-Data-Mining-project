import pandas as pd
from flask import Flask, render_template, request
import numpy as np
from collections import defaultdict

app = Flask(__name__)

# Load and preprocess data
df = pd.read_csv('degree_records.csv')

# Calculate statistics for anomaly detection
def calculate_statistics():
    # Institution-degree stats
    inst_deg_stats = df.groupby(['institution', 'degree']).agg({
        'percentage': ['mean', 'std', 'count'],
        'year': ['min', 'max']
    }).reset_index()
    inst_deg_stats.columns = ['institution', 'degree', 'mean_pct', 'std_pct', 'count', 'min_year', 'max_year']
    
    # Degree-field combinations
    degree_fields = df.groupby('degree')['field'].unique().to_dict()
    
    return {
        'inst_deg_stats': inst_deg_stats.to_dict('records'),
        'degree_fields': degree_fields,
        'earliest_year': df['year'].min(),
        'latest_year': df['year'].max()
    }

stats = calculate_statistics()

def detect_anomalies(record):
    warnings = []
    
    # Check institution-degree combination
    inst_deg_match = [x for x in stats['inst_deg_stats'] 
                     if x['institution'] == record['institution'] 
                     and x['degree'] == record['degree']]
    
    if not inst_deg_match:
        warnings.append(f"⚠️ {record['institution']} doesn't typically offer {record['degree']}")
    else:
        stats_row = inst_deg_match[0]
        # Check percentage anomalies
        if not pd.isna(stats_row['std_pct']) and stats_row['std_pct'] > 0:
            z_score = (record['percentage'] - stats_row['mean_pct']) / stats_row['std_pct']
            if abs(z_score) > 2:
                warnings.append(f"⚠️ Unusual percentage ({record['percentage']}). Typical range: {stats_row['mean_pct']:.1f}±{stats_row['std_pct']:.1f}")
        
        # Check year validity
        if record['year'] < stats_row['min_year']:
            warnings.append(f"⚠️ Suspicious year. Earliest record for this degree: {stats_row['min_year']}")
        if record['year'] > stats['latest_year']:
            warnings.append("⚠️ Future graduation year detected")
    
    # Check field-degree consistency
    if record['degree'] in stats['degree_fields']:
        if record['field'] not in stats['degree_fields'][record['degree']]:
            warnings.append(f"⚠️ Uncommon field '{record['field']}' for {record['degree']}")
    
    # Check for duplicate names with different degrees
    name_matches = df[df['name'].str.lower() == record['name'].lower()]
    if not name_matches.empty and record['degree'] not in name_matches['degree'].unique():
        warnings.append(f"⚠️ Name matches existing records with different degrees")
    
    return warnings if warnings else ["✅ No anomalies detected"]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        record = {
            'name': request.form['name'],
            'degree': request.form['degree'],
            'field': request.form['field'],
            'institution': request.form['institution'],
            'year': int(request.form['year']),
            'percentage': float(request.form['percentage'])
        }
        results = detect_anomalies(record)
        return render_template('result.html', record=record, results=results)
    
    return render_template('index.html', 
                         degrees=sorted(df['degree'].unique()),
                         institutions=sorted(df['institution'].unique()),
                         fields=sorted(df['field'].unique()))

@app.route('/verify_all')
def verify_all():
    verified_records = []
    for _, row in df.iterrows():
        record = row.to_dict()
        verified_records.append({
            'record': record,
            'warnings': detect_anomalies(record)
        })
    return render_template('verify_all.html', records=verified_records)

if __name__ == '__main__':
    app.run(debug=True)