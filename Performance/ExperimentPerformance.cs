using System.Collections.Generic;
using System.IO;

namespace Classification.Performance
{
    public class ExperimentPerformance : IComparer<ExperimentPerformance>
    {
        private readonly List<Performance> _results;
        private bool _containsDetails = true;
        private bool _classification = true;

        /**
         * <summary> A constructor which creates a new {@link List} of {@link Performance} as results.</summary>
         */
        public ExperimentPerformance()
        {
            _results = new List<Performance>();
        }

        /**
         * <summary> A constructor that takes a file name as an input and takes the inputs from that file assigns these inputs to the errorRate
         * and adds them to the results {@link List} as a new {@link Performance}.</summary>
         *
         * <param name="fileName">string input.</param>
         * @throws FileNotFoundException if a pathname fails.
         */
        public ExperimentPerformance(string fileName)
        {
            _results = new List<Performance>();
            _containsDetails = false;

            StreamReader streamReader = new StreamReader(fileName);
            while (!streamReader.EndOfStream)
            {
                var performance = streamReader.ReadLine();
                _results.Add(new Performance(double.Parse(performance)));
            }
        }

        /**
         * <summary> The add method takes a {@link Performance} as an input and adds it to the results {@link List}.</summary>
         *
         * <param name="performance">{@link Performance} input.</param>
         */
        public void Add(Performance performance)
        {
            if (!(performance is DetailedClassificationPerformance))
            {
                _containsDetails = false;
            }

            if (!(performance is ClassificationPerformance))
            {
                _classification = false;
            }

            _results.Add(performance);
        }

        /**
         * <summary> The numberOfExperiments method returns the size of the results {@link List}.</summary>
         *
         * <returns>The results {@link List}.</returns>
         */
        public int NumberOfExperiments()
        {
            return _results.Count;
        }

        /**
         * <summary> The getErrorRate method takes an index as an input and returns the errorRate at given index of results {@link List}.</summary>
         *
         * <param name="index">Index of results {@link List} to retrieve.</param>
         * <returns>The errorRate at given index of results {@link List}.</returns>
         */
        public double GetErrorRate(int index)
        {
            return _results[index].GetErrorRate();
        }

        /**
         * <summary> The getAccuracy method takes an index as an input. It returns the accuracy of a {@link Performance} at given index of results {@link List}.</summary>
         *
         * <param name="index">Index of results {@link List} to retrieve.</param>
         * <returns>The accuracy of a {@link Performance} at given index of results {@link List}.</returns>
         * @throws ClassificationAlgorithmExpectedException returns "Classification Algorithm required for accuracy metric" string.
         */
        public double GetAccuracy(int index)
        {
            if (_results[index] is ClassificationPerformance)
            {
                return ((ClassificationPerformance) _results[index]).GetAccuracy();
            }

            return 0.0;
        }

        /**
         * <summary> The meanPerformance method loops through the performances of results {@link List} and sums up the errorRates of each then
         * returns a new {@link Performance} with the mean of that summation.</summary>
         *
         * <returns>A new {@link Performance} with the mean of the summation of errorRates.</returns>
         */
        public Performance MeanPerformance()
        {
            double sumError = 0;
            foreach (var performance in _results)
            {
                sumError += performance.GetErrorRate();
            }

            return new Performance(sumError / _results.Count);
        }

        /**
         * <summary> The meanClassificationPerformance method loops through the performances of results {@link List} and sums up the accuracy of each
         * classification performance, then returns a new classificationPerformance with the mean of that summation.</summary>
         *
         * <returns>A new classificationPerformance with the mean of that summation.</returns>
         */
        public ClassificationPerformance MeanClassificationPerformance()
        {
            if (_results.Count == 0 || !_classification)
            {
                return null;
            }

            double sumAccuracy = 0;
            foreach (var performance in _results)
            {
                var classificationPerformance = (ClassificationPerformance) performance;
                sumAccuracy += classificationPerformance.GetAccuracy();
            }

            return new ClassificationPerformance(sumAccuracy / _results.Count);
        }

        /**
         * <summary> The meanDetailedPerformance method gets the first confusion matrix of results {@link List}.
         * Then, it adds new confusion matrices as the {@link DetailedClassificationPerformance} of
         * other elements of results List' confusion matrices as a {@link DetailedClassificationPerformance}.</summary>
         *
         * <returns>A new {@link DetailedClassificationPerformance} with the {@link ConfusionMatrix} sum.</returns>
         */
        public DetailedClassificationPerformance MeanDetailedPerformance()
        {
            if (_results.Count == 0 || !_containsDetails)
            {
                return null;
            }

            var sum = ((DetailedClassificationPerformance) _results[0]).GetConfusionMatrix();
            for (var i = 1; i < _results.Count; i++)
            {
                sum.AddConfusionMatrix(((DetailedClassificationPerformance) _results[i]).GetConfusionMatrix());
            }

            return new DetailedClassificationPerformance(sum);
        }

        /**
         * <summary> The standardDeviationPerformance method loops through the {@link Performance}s of results {@link List} and returns
         * a new Performance with the standard deviation.</summary>
         *
         * <returns>A new Performance with the standard deviation.</returns>
         */
        public Performance StandardDeviationPerformance()
        {
            double sumErrorRate = 0;
            var averagePerformance = MeanPerformance();
            foreach (var performance in _results)
            {
                sumErrorRate += System.Math.Pow(performance.GetErrorRate() - averagePerformance.GetErrorRate(), 2);
            }

            return new Performance(System.Math.Sqrt(sumErrorRate / (_results.Count - 1)));
        }

        /**
         * <summary> The standardDeviationClassificationPerformance method loops through the {@link Performance}s of results {@link List} and
         * returns a new {@link ClassificationPerformance} with standard deviation.</summary>
         *
         * <returns>A new {@link ClassificationPerformance} with standard deviation.</returns>
         */
        public ClassificationPerformance StandardDeviationClassificationPerformance()
        {
            if (_results.Count == 0 || !_classification)
            {
                return null;
            }

            double sumAccuracy = 0, sumErrorRate = 0;
            var averageClassificationPerformance = MeanClassificationPerformance();
            foreach (var performance in _results)
            {
                var classificationPerformance = (ClassificationPerformance) performance;
                sumAccuracy +=
                    System.Math.Pow(
                        classificationPerformance.GetAccuracy() - averageClassificationPerformance.GetAccuracy(), 2);
                sumErrorRate +=
                    System.Math.Pow(
                        classificationPerformance.GetErrorRate() - averageClassificationPerformance.GetErrorRate(), 2);
            }

            return new ClassificationPerformance(System.Math.Sqrt(sumAccuracy / (_results.Count - 1)), System.Math.Sqrt(
                sumErrorRate / (
                    _results.Count - 1)));
        }

        /**
         * <summary> The isBetter method  takes an {@link ExperimentPerformance} as an input and returns true if the result of compareTo method is positive
         * and false otherwise.</summary>
         *
         * <param name="experimentPerformance">{@link ExperimentPerformance} input.</param>
         * <returns>True if the result of compareTo method is positive and false otherwise.</returns>
         */
        public bool IsBetter(ExperimentPerformance experimentPerformance)
        {
            return Compare(this, experimentPerformance) > 0;
        }


        public int Compare(ExperimentPerformance x, ExperimentPerformance y)
        {
            if (x.MeanClassificationPerformance().GetAccuracy() < y.MeanClassificationPerformance().GetAccuracy())
            {
                return -1;
            }

            if (x.MeanClassificationPerformance().GetAccuracy() > y.MeanClassificationPerformance().GetAccuracy())
            {
                return 1;
            }

            return 0;
        }
    }
}