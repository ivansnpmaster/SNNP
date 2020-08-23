using System;
using System.Collections.Generic;

namespace SNNP.kMeans
{
    [Serializable()]
    public class Cluster
    {
        public double[] position;
        public List<double[]> points = new List<double[]>();

        public Cluster(double[] _position)
        {
            position = _position;
            points.Add(position);
        }

        public double[] RecalculatePosition()
        {
            double[,] p = new double[points.Count, position.Length];

            for (int i = 0; i < p.GetLength(0); i++)
                for (int j = 0; j < p.GetLength(1); j++)
                    p[i, j] = points[i][j];

            position = Mathf.GetCentroid(p);

            return position;
        }
    }
}