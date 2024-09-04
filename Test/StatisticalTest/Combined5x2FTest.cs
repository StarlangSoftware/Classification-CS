using Classification.DistanceMetric;
using Classification.Experiment;
using Classification.Model;
using Classification.Model.DecisionTree;
using Classification.Parameter;
using Classification.Performance;
using Classification.StatisticalTest;
using NUnit.Framework;
using Test.Classifier;

namespace Test.StatisticalTest
{
    public class Combined5x2FTest : ClassifierTest
    {
        [Test]
        public void TestCompare()
        {
            var mxKFoldRun = new MxKFoldRun(5, 2);
            var experimentPerformance1 =
                mxKFoldRun.Execute(new Classification.Experiment.Experiment(new DecisionTree(), new C45Parameter(1, true, 0.2), iris));
            var experimentPerformance2 = mxKFoldRun.Execute(new Classification.Experiment.Experiment(new LinearPerceptronModel(),
                new LinearPerceptronParameter(1, 0.1, 0.99, 0.2, 100), iris));
            var combined5x2F = new Combined5x2F();
            Assert.AreEqual(0.734, combined5x2F.Compare(experimentPerformance1, experimentPerformance2).GetPValue(),
                0.001);
            experimentPerformance1 =
                mxKFoldRun.Execute(new Classification.Experiment.Experiment(new DecisionTree(), new C45Parameter(1, true, 0.2), tictactoe));
            experimentPerformance2 =
                mxKFoldRun.Execute(new Classification.Experiment.Experiment(new BaggingModel(), new BaggingParameter(1, 50), tictactoe));
            Assert.AreEqual(0.003, combined5x2F.Compare(experimentPerformance1, experimentPerformance2).GetPValue(),
                0.001);
            experimentPerformance1 = mxKFoldRun.Execute(new Classification.Experiment.Experiment(new LdaModel(), new Parameter(1), dermatology));
            experimentPerformance2 = mxKFoldRun.Execute(new Classification.Experiment.Experiment(new LinearPerceptronModel(),
                new LinearPerceptronParameter(1, 0.1, 0.99, 0.2, 100), dermatology));
            Assert.AreEqual(0.647, combined5x2F.Compare(experimentPerformance1, experimentPerformance2).GetPValue(),
                0.001);
            experimentPerformance1 = mxKFoldRun.Execute(new Classification.Experiment.Experiment(new DummyModel(), new Parameter(1), nursery));
            experimentPerformance2 = mxKFoldRun.Execute(new Classification.Experiment.Experiment(new NaiveBayesModel(), new Parameter(1), nursery));
            Assert.AreEqual(0.0, combined5x2F.Compare(experimentPerformance1, experimentPerformance2).GetPValue(), 0.0001);
            experimentPerformance1 = mxKFoldRun.Execute(new Classification.Experiment.Experiment(new NaiveBayesModel(), new Parameter(1), car));
            experimentPerformance2 =
                mxKFoldRun.Execute(new Classification.Experiment.Experiment(new BaggingModel(), new BaggingParameter(1, 50), car));
            Assert.AreEqual(0.008, combined5x2F.Compare(experimentPerformance1, experimentPerformance2).GetPValue(),
                0.001);
            experimentPerformance1 =
                mxKFoldRun.Execute(new Classification.Experiment.Experiment(new KnnModel(), new KnnParameter(1, 3, new EuclidianDistance()), bupa));
            experimentPerformance2 = mxKFoldRun.Execute(new Classification.Experiment.Experiment(new LdaModel(), new Parameter(1), bupa));
            Assert.AreEqual(0.264, combined5x2F.Compare(experimentPerformance1, experimentPerformance2).GetPValue(),
                0.001);
        }
    }
}