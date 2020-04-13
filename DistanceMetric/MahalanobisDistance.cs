using Math;

namespace Classification.DistanceMetric
{
    public class MahalanobisDistance : DistanceMetric
    {
        private readonly Matrix _covarianceInverse;

        /**
         * <summary> Constructor for the MahalanobisDistance class. Basically sets the inverse of the covariance matrix.</summary>
         *
         * <param name="covarianceInverse">Inverse of the covariance matrix.</param>
         */
        public MahalanobisDistance(Matrix covarianceInverse)
        {
            this._covarianceInverse = covarianceInverse;
        }

        /**
         * <summary> Calculates Mahalanobis distance between two instances. (x^(1) - x^(2)) S (x^(1) - x^(2))^T</summary>
         *
         * <param name="instance1">First instance.</param>
         * <param name="instance2">Second instance.</param>
         * <returns>Mahalanobis distance between two instances.</returns>
         */
        public double Distance(Instance.Instance instance1, Instance.Instance instance2)
        {
            var v1 = instance1.ToVector();
            var v2 = instance2.ToVector();
            v1.Subtract(v2);
            var v3 = _covarianceInverse.MultiplyWithVectorFromLeft(v1);
            return v3.DotProduct(v1);
        }
    }
}