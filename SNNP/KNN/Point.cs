namespace SNNP.KNN
{
    public class Point
    {
        public double[] f;
        public string label;

        public Point(object[] data)
        {
            for (int i = 0; i < data.Length - 1; i++)
                f[i] = (double)data[i];

            label = data[data.Length - 1].ToString();
        }
    }
}