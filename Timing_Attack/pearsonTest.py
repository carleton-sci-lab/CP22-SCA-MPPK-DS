import xml.etree.ElementTree as ET

# Parse the XML file
tree = ET.parse('hm_timing_data2.xml')
root = tree.getroot()

# Initialize the data and labels lists
data = []
labels = ['mu']
for i in range(10):
    labels.append('hm{}'.format(i))

# Loop through each "row" element in the XML file
for row in root.findall('row'):
    # Extract the mu value and add it to the data list
    mu = int(row.find('mu').text)
    row_data = [mu]

    # Extract the hm values and add them to the row_data list
    for i in range(10):
        hm = float(row.find('hm{}'.format(i)).text)
        row_data.append(hm)

    # Add the row_data list to the data list
    data.append(row_data)

# Calculate the correlation coefficients
correlations = []
for i in range(1, len(labels)):
    column1 = [row[i] for row in data]
    for j in range(i+1, len(labels)):
        column2 = [row[j] for row in data]
        correlation = pearsonr(column1, column2)
        correlations.append((labels[i], labels[j], correlation[0], correlation[1]))

# Print the results
for correlation in correlations:
    print('{} vs {}: r={}, p={}'.format(correlation[0], correlation[1], correlation[2], correlation[3]))
