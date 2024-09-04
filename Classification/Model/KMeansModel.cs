using System.IO;
using Classification.DistanceMetric;
using Classification.Parameter;

namespace Classification.Model
{
    public class KMeansModel : GaussianModel
    {
        private InstanceList.InstanceList _classMeans;
        private DistanceMetric.DistanceMetric _distanceMetric;

        public KMeansModel()
        {
        }

        /// <summary>
        /// Loads a K-means model from an input model file.
        /// </summary>
        /// <param name="fileName">Model file name.</param>
        private void Load(string fileName)
        {
            _distanceMetric = new EuclidianDistance();
            var input = new StreamReader(fileName);
            LoadPriorDistribution(input);
            _classMeans = LoadInstanceList(input);
            input.Close();
        }

        /// <summary>
        /// Loads a K-means model from an input model file.
        /// </summary>
        /// <param name="fileName">Model file name.</param>
        public KMeansModel(string fileName)
        {
            Load(fileName);
        }

        /**
         * <summary> The calculateMetric method takes an {@link Instance} and a String as inputs. It loops through the class means, if
         * the corresponding class label is same as the given String it returns the negated distance between given instance and the
         * current item of class means. Otherwise it returns the smallest negative number.</summary>
         *
         * <param name="instance">{@link Instance} input.</param>
         * <param name="ci">      String input.</param>
         * <returns>The negated distance between given instance and the current item of class means.</returns>
         */
        protected override double CalculateMetric(Instance.Instance instance, string ci)
        {
            for (var i = 0; i < _classMeans.Size(); i++)
            {
                if (_classMeans.Get(i).GetClassLabel() == ci)
                {
                    return -_distanceMetric.Distance(instance, _classMeans.Get(i));
                }
            }

            return double.MinValue;
        }

        /**
         * <summary> Training algorithm for K-Means classifier. K-Means finds the mean of each class for training.</summary>
         *
         * <param name="trainSet">  Training data given to the algorithm.</param>
         * <param name="parameters">distanceMetric: distance metric used to calculate the distance between two instances.</param>
         */
        public override void Train(InstanceList.InstanceList trainSet, Parameter.Parameter parameters)
        {
            priorDistribution = trainSet.ClassDistribution();
            _classMeans = new InstanceList.InstanceList();
            var classLists = trainSet.DivideIntoClasses();
            for (var i = 0; i < classLists.Size(); i++)
            {
                _classMeans.Add(classLists.Get(i).Average());
            }

            _distanceMetric = ((KMeansParameter)parameters).GetDistanceMetric();
        }

        /// <summary>
        /// Loads the K-means model from an input file.
        /// </summary>
        /// <param name="fileName">File name of the K-means model.</param>
        public override void LoadModel(string fileName)
        {
            Load(fileName);
        }
    }
}