using Classification.DistanceMetric;

namespace Classification.Parameter
{
    public class KMeansParameter : Parameter
    {
        protected readonly DistanceMetric.DistanceMetric distanceMetric;

        /**
         * <summary> Parameters of the K Means classifier.</summary>
         *
         * <param name="seed">Seed is used for random number generation.</param>
         */
        public KMeansParameter(int seed) : base(seed)
        {
            distanceMetric = new EuclidianDistance();
        }

        /**
         * <summary> Parameters of the K Means classifier.</summary>
         *
         * <param name="seed">          Seed is used for random number generation.</param>
         * <param name="distanceMetric">distance metric used to calculate the distance between two instances.</param>
         */
        public KMeansParameter(int seed, DistanceMetric.DistanceMetric distanceMetric) : base(seed)
        {
            this.distanceMetric = distanceMetric;
        }

        /**
         * <summary> Accessor for the distanceMetric.</summary>
         *
         * <returns>The distanceMetric.</returns>
         */
        public DistanceMetric.DistanceMetric GetDistanceMetric()
        {
            return distanceMetric;
        }
    }
}