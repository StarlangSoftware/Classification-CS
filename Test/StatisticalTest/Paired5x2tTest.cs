using Classification.Classifier;
using Classification.DistanceMetric;
using Classification.Experiment;
using Classification.Parameter;
using Classification.StatisticalTest;
using NUnit.Framework;
using Test.Classifier;

namespace Test.StatisticalTest
{
    public class Paired5x2tTest : ClassifierTest
    {
        [Test]
        public void TestCompare()
        {
            var mxKFoldRun = new MxKFoldRun(5, 2);
            var experimentPerformance1 =
                mxKFoldRun.Execute(new Classification.Experiment.Experiment(new C45(), new C45Parameter(1, true, 0.2), iris));
            var experimentPerformance2 = mxKFoldRun.Execute(new Classification.Experiment.Experiment(new LinearPerceptron(),
                new LinearPerceptronParameter(1, 0.1, 0.99, 0.2, 100), iris));
            var paired5x2t = new Paired5x2t();
            Assert.AreEqual(0.892, paired5x2t.Compare(experimentPerformance1, experimentPerformance2).GetPValue(), 0.001);
            experimentPerformance1 =
                mxKFoldRun.Execute(new Classification.Experiment.Experiment(new C45(), new C45Parameter(1, true, 0.2), tictactoe));
            experimentPerformance2 =
                mxKFoldRun.Execute(new Classification.Experiment.Experiment(new Bagging(), new BaggingParameter(1, 50), tictactoe));
            Assert.AreEqual(0.108, paired5x2t.Compare(experimentPerformance1, experimentPerformance2).GetPValue(),
                0.001);
            experimentPerformance1 = mxKFoldRun.Execute(new Classification.Experiment.Experiment(new Lda(), new Parameter(1), dermatology));
            experimentPerformance2 = mxKFoldRun.Execute(new Classification.Experiment.Experiment(new LinearPerceptron(),
                new LinearPerceptronParameter(1, 0.1, 0.99, 0.2, 100), dermatology));
            Assert.AreEqual(0.892, paired5x2t.Compare(experimentPerformance1, experimentPerformance2).GetPValue(),
                0.001);
            experimentPerformance1 = mxKFoldRun.Execute(new Classification.Experiment.Experiment(new Dummy(), new Parameter(1), nursery));
            experimentPerformance2 = mxKFoldRun.Execute(new Classification.Experiment.Experiment(new NaiveBayes(), new Parameter(1), nursery));
            Assert.AreEqual(0.0019, paired5x2t.Compare(experimentPerformance1, experimentPerformance2).GetPValue(), 0.0001);
            experimentPerformance1 = mxKFoldRun.Execute(new Classification.Experiment.Experiment(new NaiveBayes(), new Parameter(1), car));
            experimentPerformance2 =
                mxKFoldRun.Execute(new Classification.Experiment.Experiment(new Bagging(), new BaggingParameter(1, 50), car));
            Assert.AreEqual(0.2578, paired5x2t.Compare(experimentPerformance1, experimentPerformance2).GetPValue(),
                0.0001);
            experimentPerformance1 =
                mxKFoldRun.Execute(new Classification.Experiment.Experiment(new Knn(), new KnnParameter(1, 3, new EuclidianDistance()), bupa));
            experimentPerformance2 = mxKFoldRun.Execute(new Classification.Experiment.Experiment(new Lda(), new Parameter(1), bupa));
            Assert.AreEqual(0.350, paired5x2t.Compare(experimentPerformance1, experimentPerformance2).GetPValue(), 0.001);
        }
    }
}