namespace Classification.Parameter
{
    public class KnnParameter : KMeansParameter
    {
        private readonly int _k;

        /**
         * <summary> Parameters of the K-nearest neighbor classifier.</summary>
         *
         * <param name="seed">          Seed is used for random number generation.</param>
         * <param name="k">             Parameter of the K-nearest neighbor algorithm.</param>
         * <param name="distanceMetric">Used to calculate the distance between two instances.</param>
         */
        public KnnParameter(int seed, int k, DistanceMetric.DistanceMetric distanceMetric) : base(seed, distanceMetric)
        {
            this._k = k;
        }

        /**
         * <summary> Parameters of the K-nearest neighbor classifier.</summary>
         *
         * <param name="seed">          Seed is used for random number generation.</param>
         * <param name="k">             Parameter of the K-nearest neighbor algorithm.</param>
         */
        public KnnParameter(int seed, int k) : base(seed)
        {
            this._k = k;
        }

        /**
         * <summary> Accessor for the k.</summary>
         *
         * <returns>Value of the k.</returns>
         */
        public int GetK()
        {
            return _k;
        }
    }
}