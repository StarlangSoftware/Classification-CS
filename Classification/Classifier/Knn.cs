using Classification.Model;
using Classification.Parameter;

namespace Classification.Classifier
{
    public class Knn : Classifier
    {
        /**
         * <summary> Training algorithm for K-nearest neighbor classifier.</summary>
         *
         * <param name="trainSet">  Training data given to the algorithm.</param>
         * <param name="parameters">K: k parameter of the K-nearest neighbor algorithm
         *                   distanceMetric: distance metric used to calculate the distance between two instances.</param>
         */
        public override void Train(InstanceList.InstanceList trainSet, Parameter.Parameter parameters)
        {
            model = new KnnModel(trainSet, ((KnnParameter) parameters).GetK(),
                ((KnnParameter) parameters).GetDistanceMetric());
        }

        /// <summary>
        /// Loads the K-nearest neighbor model from an input file.
        /// </summary>
        /// <param name="fileName">File name of the K-nearest neighbor model.</param>
        public override void LoadModel(string fileName)
        {
            model = new KnnModel(fileName);
        }
    }
}