# Customer Segmentation with K-means Clustering

This project aims to perform customer segmentation using the K-means clustering algorithm. Customer segmentation helps identify groups of customers with similar characteristics, allowing businesses to better understand their customer base and tailor their marketing strategies accordingly.

## Dataset

The dataset used for this project contains customer information, such asSex, Marital status, Age, Education, Income Occupation etc. The dataset should be preprocessed and cleaned before applying the K-means algorithm. Some common preprocessing steps include handling missing values, scaling numerical features, and encoding categorical variables.

## Dependencies

The following libraries are required to run the code:

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## Usage

1. Clone the repository:

```
git clone https://github.com/your-username/customer-segmentation.git
cd customer-segmentation
```

2. Install the required dependencies:

```
pip install -r requirements.txt
```

3. Preprocess the data:

   - Place the raw customer data file (`customers.csv`) in the `data/` directory.
   - Run the preprocessing script to clean and transform the data:

   ```
   python preprocessing.py
   ```

   The preprocessed data will be saved as `preprocessed_data.csv`.

4. Perform customer segmentation:

   - Run the K-means clustering script:

   ```
   python kmeans_clustering.py
   ```

   The script applies the K-means algorithm on the preprocessed data and generates customer segments. The resulting segments will be saved as `customer_segments.csv`.

5. Analyze and interpret the results:

   - Use the generated `customer_segments.csv` file to analyze the characteristics of each customer segment.
   - Visualize the clusters and explore the relationships between different customer attributes using plots and statistical analysis.


## Acknowledgments

- [Scikit-learn](https://scikit-learn.org) - Python machine learning library.
- [Pandas](https://pandas.pydata.org) - Data manipulation and analysis library.
- [Matplotlib](https://matplotlib.org) - Data visualization library.

