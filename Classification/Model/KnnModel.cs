using System.Collections.Generic;
using System.IO;
using Classification.DistanceMetric;
using Classification.Instance;

namespace Classification.Model
{
    public class KnnModel : Model
    {
        private readonly InstanceList.InstanceList _data;
        private readonly int _k;
        private readonly DistanceMetric.DistanceMetric _distanceMetric;

        /**
         * <summary> Constructor that sets the data {@link InstanceList}, k value and the {@link DistanceMetric}.</summary>
         *
         * <param name="data">          {@link InstanceList} input.</param>
         * <param name="k">             K value.</param>
         * <param name="distanceMetric">{@link DistanceMetric} input.</param>
         */
        public KnnModel(InstanceList.InstanceList data, int k, DistanceMetric.DistanceMetric distanceMetric)
        {
            _data = data;
            _k = k;
            _distanceMetric = distanceMetric;
        }

        /// <summary>
        /// Loads a K-nearest neighbor model from an input model file.
        /// </summary>
        /// <param name="fileName">Model file name.</param>
        public KnnModel(string fileName)
        {
            _distanceMetric = new EuclidianDistance();
            var input = new StreamReader(fileName);
            _k = int.Parse(input.ReadLine());
            _data = LoadInstanceList(input);
            input.Close();
        }

        /**
         * <summary> The predict method takes an {@link Instance} as an input and finds the nearest neighbors of given instance. Then
         * it returns the first possible class label as the predicted class.</summary>
         *
         * <param name="instance">{@link Instance} to make prediction.</param>
         * <returns>The first possible class label as the predicted class.</returns>
        */
        public override string Predict(Instance.Instance instance)
        {
            var nearestNeighbors = NearestNeighbors(instance);
            string predictedClass;
            if (instance is CompositeInstance compositeInstance && nearestNeighbors.Size() == 0) {
                predictedClass = compositeInstance.GetPossibleClassLabels()[0];
            } else {
                predictedClass = Classifier.Classifier.GetMaximum(nearestNeighbors.GetClassLabels());
            }
            return predictedClass;
        }

        /// <summary>
        /// Calculates the posterior probability distribution for the given instance according to K-means model.
        /// </summary>
        /// <param name="instance">Instance for which posterior probability distribution is calculated.</param>
        /// <returns>Posterior probability distribution for the given instance.</returns>
        public override Dictionary<string, double> PredictProbability(Instance.Instance instance)
        {
            var nearestNeighbors = NearestNeighbors(instance);
            return nearestNeighbors.ClassDistribution().GetProbabilityDistribution();
        }

        /**
         * <summary> The nearestNeighbors method takes an {@link Instance} as an input. First it gets the possible class labels, then loops
         * through the data {@link InstanceList} and creates new {@link ArrayList} of {@link KnnInstance}s and adds the corresponding data with
         * the distance between data and given instance. After sorting this newly created ArrayList, it loops k times and
         * returns the first k instances as an {@link InstanceList}.</summary>
         *
         * <param name="instance">{@link Instance} to find nearest neighbors/</param>
         * <returns>The first k instances which are nearest to the given instance as an {@link InstanceList}.</returns>
         */
        public InstanceList.InstanceList NearestNeighbors(Instance.Instance instance)
        {
            var result = new InstanceList.InstanceList();
            var instances = new List<KnnInstance>();
            List<string> possibleClassLabels = null;
            if (instance is CompositeInstance compositeInstance) {
                possibleClassLabels = compositeInstance.GetPossibleClassLabels();
            }
            for (var i = 0; i < _data.Size(); i++)
            {
                if (!(instance is CompositeInstance) || possibleClassLabels.Contains(
                    _data.Get(i).GetClassLabel())) {
                    instances.Add(new KnnInstance(_data.Get(i), _distanceMetric.Distance(_data.Get(i), instance)));
                }
            }

            instances.Sort(new KnnInstanceComparator());
            for (var i = 0; i < System.Math.Min(_k, instances.Count); i++)
            {
                result.Add(instances[i].Instance);
            }

            return result;
        }


        private class KnnInstance
        {
            internal readonly double Distance;
            internal readonly Instance.Instance Instance;

            /**
             * The constructor that sets the instance and distance value.
             *
             * <param name="instance">{@link Instance} input.</param>
             * <param name="distance">Double distance value.</param>
             */
            internal KnnInstance(Instance.Instance instance, double distance)
            {
                this.Instance = instance;
                this.Distance = distance;
            }

            /**
             * The toString method returns the concatenation of class label of the instance and the distance value.
             *
             * <returns>The concatenation of class label of the instance and the distance value.</returns>
             */
            public override string ToString()
            {
                var str = "";
                str += Instance.GetClassLabel() + " " + Distance;
                return str;
            }
        }


        private class KnnInstanceComparator : IComparer<KnnInstance> {
            /**
             * The compare method takes two {@link KnnInstance}s as inputs and returns -1 if the distance of first instance is
             * less than the distance of second instance, 1 if the distance of first instance is greater than the distance of second instance,
             * and 0 if they are equal to each other.
             *
             * <param name="instance1">First {@link KnnInstance} to compare.</param>
             * <param name="instance2">Second {@link KnnInstance} to compare.</param>
             * <returns>-1 if the distance of first instance is less than the distance of second instance,</returns>
             * 1 if the distance of first instance is greater than the distance of second instance,
             * 0 if they are equal to each other.
             */
            public int Compare(KnnInstance instance1, KnnInstance instance2)
            {
                if (instance1.Distance < instance2.Distance)
                {
                    return -1;
                }
                else
                {
                    if (instance1.Distance > instance2.Distance)
                    {
                        return 1;
                    }
                    else
                    {
                        return 0;
                    }
                }
            }
        }
    }
}