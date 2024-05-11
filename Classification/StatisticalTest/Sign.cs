using Classification.Performance;

namespace Classification.StatisticalTest
{
    public class Sign : PairedTest
    {
        /// <summary>
        /// Calculates n!.
        /// </summary>
        /// <param name="n">n is n!</param>
        /// <returns>n!.</returns>
        private int Factorial(int n)
        {
            int i, result = 1;
            for (i = 2; i <= n; i++)
                result *= i;
            return result;
        }

        /// <summary>
        /// Calculates m of n that is C(n, m)
        /// </summary>
        /// <param name="m">m in C(m, n)</param>
        /// <param name="n">n in C(m, n)</param>
        /// <returns>C(m, n)</returns>
        private int Binomial(int m, int n)
        {
            if (n == 0 || m == n)
                return 1;
            return Factorial(m) / (Factorial(n) * Factorial(m - n));
        }

        /// <summary>
        /// Compares two classification algorithms based on their performances (accuracy or error rate) using sign test.
        /// </summary>
        /// <param name="classifier1">Performance (error rate or accuracy) results of the first classifier.</param>
        /// <param name="classifier2">Performance (error rate or accuracy) results of the second classifier.</param>
        /// <returns>Statistical test result of the comparison.</returns>
        public override StatisticalTestResult Compare(ExperimentPerformance classifier1,
            ExperimentPerformance classifier2)
        {
            int plus = 0, minus = 0;
            for (var i = 0; i < classifier1.NumberOfExperiments(); i++)
            {
                if (classifier1.GetErrorRate(i) < classifier2.GetErrorRate(i))
                {
                    plus++;
                }
                else
                {
                    if (classifier1.GetErrorRate(i) > classifier2.GetErrorRate(i))
                    {
                        minus++;
                    }
                }
            }

            var total = plus + minus;

            var pValue = 0.0;
            for (var i = 0; i <= plus; i++)
            {
                pValue += Binomial(total, i) / System.Math.Pow(2, total);
            }

            return new StatisticalTestResult(pValue, false);
        }
    }
}