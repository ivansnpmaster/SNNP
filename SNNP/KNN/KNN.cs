using System;
using SNNP.MLP;
using System.Linq;
using System.Collections.Generic;

namespace SNNP.KNN
{
    public class KNN
    {
        Point[] points;
        int _k;

        int rows;
        int cols;

        /// <summary>
        /// Setup an implementation of vanilla KNN algorithm.
        /// </summary>
        /// <param name="dataset">Every row in the dataset must have the label as the last array element.</param>
        /// <param name="k">Number of neighbors.</param>
        public KNN(object[,] dataset, int k = 1)
        {
            rows = dataset.GetLength(0);
            cols = dataset.GetLength(1);
            points = new Point[rows];

            for (int i = 0; i < rows; i++)
            {
                object[] features = new object[cols];

                for (int j = 0; j < cols; j++)
                    features[j] = dataset[i, j];

                points[i] = new Point(features);
            }

            _k = k;
        }

        /// <summary>
        /// Classify a given input according to the setup dataset provided.
        /// </summary>
        /// <param name="input">Features/input containing data.</param>
        /// <returns>A label for the input.</returns>
        public string Classify(double[] input)
        {
            try
            {
                Dictionary<int, double> distances = new Dictionary<int, double>();

                for (int i = 0; i < rows; i++)
                    distances.Add(i, Mathf.GetDistanceSqr(input, points[i].f));

                // Starting index, number of elements to return (k)
                var knn = distances.OrderBy(d => d.Value).ToList().GetRange(0, _k);

                Dictionary<string, int> labels = new Dictionary<string, int>();

                foreach (var n in knn)
                {
                    string label = points[n.Key].label;

                    if (labels.ContainsKey(label))
                        labels[label]++;
                    else
                        labels.Add(label, 1);
                }

                return labels.OrderByDescending(l => l.Value).ToList()[0].Key;
            }
            catch (Exception) { throw; }
        }
    }
}