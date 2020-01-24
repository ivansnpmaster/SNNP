using SNNP.MLP;
using System.Linq;
using System.Collections.Generic;

namespace SNNP.KNN
{
    public class KNN
    {
        Point[] points;
        int _k;

        /// <summary>
        /// Setup an implementation of vanilla KNN algorithm.
        /// </summary>
        /// <param name="dataset">Every row in the dataset must have the label as the last array element.</param>
        /// <param name="k">Number of neighbors.</param>
        public KNN(object[,] dataset, int k = 1)
        {
            points = new Point[dataset.GetLength(0)];

            for (int i = 0; i < dataset.GetLength(0); i++)
            {
                object[] features = new object[dataset.GetLength(1)];

                for (int j = 0; j < features.Length; j++)
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
            Dictionary<int, double> distances = new Dictionary<int, double>();

            for (int i = 0; i < points.Length; i++)
                distances.Add(i, Utility.GetDistance(input, points[i].f));

            // Starting index, number of elements to return (k)
            var knn = distances.OrderBy(d => d.Value).ToList().GetRange(0, _k);

            Dictionary<string, int> labels = new Dictionary<string, int>();

            foreach(var n in knn)
            {
                string label = points[n.Key].label;

                if (labels.ContainsKey(label))
                    labels[label]++;
                else
                    labels.Add(label, 1);
            }
            
            return labels.OrderByDescending(l => l.Value).ToList()[0].Key;
        }
    }
}