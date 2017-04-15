using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;
using System.IO;

/*
 * 
 * 
 * 
 * 
 * Some notes, so I don't have to re-google these:
 * 
 * 
 *      The range of DCT for 4x4 blocks 
 *      of pixel intensities between (0,255) 
 *      is (-4080,4080) (255*4*4)
 *      
 * 
 *      The range of DCT for 4x4x4 blocks 
 *      of pixel intensities between (0,255) 
 *      is (-16320,16320) (255*4*4*4)
 *      
 * 
 *      Look at the size of the transition matrix:
 *      
 *      16,646,400 matrix entities for DCT 4x4 === (HUGE)
 *      266,342,400 matrix entities DCT 4x4 === (TOO BIG)
 *      
 *      So we use markov_t = 10
 *      and truncate coefficients at 10
 *      to govern the matrix's size 
 *      so that it doesn't grow too large
 *      our transition matrix is 10x10
 *      
 *      With [1] and markov_t =10, each group gives us 
 *      
 *      10x10 intra frame features
 *      10x10 inter frame features
 *      3 abs central moment features
 *      1 kurtosis feature
 *      1 skewness feature
 *      
 *      For a 205-dimensional feature vector
 * 
 *      [1] performs YouTube Steganalysis on a frame by frame (Group by Group) basis NOT a video by video basis.
 *          This means that the cluster representatives are not entire videos, but instead, groups (4-frames).
 *          
 *      Explained by authors:
 *      
 *                   * 4.2 Experiment Setup
 *                  As shown in [18], unsupervised classification is more
 *                  practical, even the performance is not as good as supervised
 *                  classification. In our experiments, K-means clustering,
 *                  which is a very basic and widely used clustering algorithm
 *                  with cosine distance was considered. The performance is
 *                  measured with True Positive Rate (TPR) and True Negative
 *                  Rate (TNR). TPR represents the probability of that stego is
 *                  classified into stego. And TNR means the probability of that
 *                  cover is classified into cover. One thing should be
 *                  remembered is that our proposed scheme is based on a
 *                  frame-by-frame or group-by-group method.
 * 
 * 
 *      [1] uses k-means clustering with 2 clusters, "Steg" and "Clean". 
 *      It is up to you to decide the initial clusters and k.
 *      
 * 
 *      Try to decide initial centers as the most extreme example of high activity steganography  group for "Steg", 
 *      and the smoothest low-motion video group for "Clean"
 *      
 * 
 *      The steganography algorithms that we use are different than those used in [1] but steganalysis is the same.
 *      
 *      The TPR (Total Positive Rate) detection is based on how many groups are correctly classified in the "Steg" cluster per total frames of a video
 *      The TNR (Total Negative Rate) detection is based on how many groups are correctly classified in the "Clean" cluster per total frames of a video
 *      
 *      A Markov() represents a single "Group", mentioned in the paper
 *      A Markov()[] represents a collection of "Group" objects.
 *      
 *      A Markov()[] is used in this case to represent a collection of groups in a cluster.
 
        See Main() for a full and detailed explanation of Steganalysis.
 
 
 */

namespace H264Stego
{
    class Program
    {
        static long filesizespecified;
        static bool loop = true;
        static long pos;
        static double[] acc;
        static FeatureVector[] fv;
        static FeatureVector[] fvp;

        static long pos_bit;
        static long pos_rep;
        static int tmod = 0;
        static byte[] finalFile;
        static int markov_t = 10; // as in [1]
        const int[,] RESOLUTIONS = {{176, 144}, {320, 240}, {640, 480}, {1280, 720}, {1920, 1080}};

        public class Frequency_4x4
        {
            public double[,] frequencies;

            public Frequency_4x4()
            {
                frequencies = new double[4, 4];
            }
        }

        public class CompInt : IComparer<int>
        {
            public int Compare(int a, int b)
            {
                return a - b;
            }
        }

        public class Markov
        {
            public double[, ,] markov;

            public Markov(int t1, int t2)
            {
                markov = new double[t1*2, t2*2, 3];
            }

            public Markov()
            {
                markov = new double[markov_t*2, markov_t*2, 3];
            }

        }

        [Serializable]
        public class FeatureVector
        {
            public double[] vec;
            public int id = 0;
            public FeatureVector(int size, int id)
            {
                vec = new double[size];
                this.id = id;
            }
            public FeatureVector()
            {
                vec = new double[888];
                this.id = -1;
            }
        }

        public class Frequency_4x4x4
        {
            public double[, ,] frequencies;

            public Frequency_4x4x4()
            {
                frequencies = new double[4, 4, 4];
            }
        }

        /// <summary>
        /// Generates a DCT3D "group" of 4 adjacent video frames' transformed into 4 adjacent coefficient planes, see [1]
        /// </summary>
        /// <param name="b0">First Frame</param>
        /// <param name="b1">Second Frame</param>
        /// <param name="b2">Fourth Frame</param>
        /// <param name="b3">Third Frame</param>
        /// <returns></returns>
        static double[, ,] M_Group(Bitmap b0, Bitmap b1, Bitmap b2, Bitmap b3, int mode)
        {
            double[, ,] coefficients = new double[b0.Width, b0.Height, 2];
            Frequency_4x4x4 f = new Frequency_4x4x4();
            Bitmap src0 = new Bitmap(4, 4);
            Bitmap src1 = new Bitmap(4, 4);
            Bitmap src2 = new Bitmap(4, 4);
            Bitmap src3 = new Bitmap(4, 4);

            for (int i = 0; i < b0.Height; i += 4)
                for (int j = 0; j < b0.Width; j += 4)
                {

                    src0 = b0.Clone(new Rectangle(j, i, 4, 4), b0.PixelFormat);
                    src1 = b1.Clone(new Rectangle(j, i, 4, 4), b1.PixelFormat);
                    src2 = b2.Clone(new Rectangle(j, i, 4, 4), b2.PixelFormat);
                    src3 = b3.Clone(new Rectangle(j, i, 4, 4), b3.PixelFormat);

                    //f = DCT3D_4x4x4(src0, src1, src2, src3, 0);
                    f = Faster_DCT3D_4x4x4(src0, src1, src2, src3, 0);

                    if (mode == 0)
                    for (int z = 0; z < 2; z++)
                        for (int y = 0; y < src0.Height; y++)
                            for (int x = 0; x < src0.Width; x++)
                                coefficients[j + x, i + y, z] = f.frequencies[x, y, z];

                    if(mode == 1)
                    for (int z = 0; z < 2; z++)
                         for (int y = 0; y < src0.Height; y++)
                             for (int x = 0; x < src0.Width; x++)
                                 coefficients[j + x, i + y, z] = f.frequencies[x, y, z+2];

                    src0.Dispose();
                    src1.Dispose();
                    src2.Dispose();
                    src3.Dispose();
                }

            return coefficients;

        }

        /// <summary>
        /// Alternative DCT
        /// </summary>
        /// <param name="b0"></param>
        /// <param name="b1"></param>
        /// <param name="b2"></param>
        /// <param name="b3"></param>
        /// <returns></returns>
        static Frequency_4x4x4 Fast_DCT3D_4x4x4(Bitmap b0, Bitmap b1, Bitmap b2, Bitmap b3)
        {
            int m = 2; // lb(4)
            int ns = 8; // 4 << 1
            int ns2 = 2; // 4 >> 1
            float pi = (float)Math.PI;
            //float invr1 = 0.707106781f;

            float p0;
            float p1;
            float p2;
            float p3;
            float p4;
            float p5;
            float p6;
            float p7;
            float p8;
            float p9;
            float p10;
            float p11;

            float Cfacm;


            Frequency_4x4x4 f = new Frequency_4x4x4();
            Bitmap[] b = { b0, b1, b2, b3 };

            /* 3d Re-ordering */

            for (int n1 = 0; n1 <= 1; n1++)
                for (int n2 = 0; n2 <= 1; n2++)
                    for (int n3 = 0; n3 <= 1; n3++)
                    {
                        f.frequencies[n1, n2, n3] = b[2 * n3].GetPixel(2 * n1, 2 * n2).ToArgb();
                        f.frequencies[n1, n2, 4 - n3 - 1] = b[(2 * n3) + 1].GetPixel(2 * n1, 2 * n2).ToArgb();
                        f.frequencies[n1, 4 - n2 - 1, n3] = b[2 * n3].GetPixel(2 * n1, (2 * n2) + 1).ToArgb();
                        f.frequencies[n1, 4 - n2 - 1, 4 - n3 - 1] = b[(2 * n3) + 1].GetPixel(2 * n1, (2 * n2) + 1).ToArgb();
                        f.frequencies[4 - n1 - 1, n2, n3] = b[2 * n3].GetPixel((2 * n1) + 1, 2 * n2).ToArgb();
                        f.frequencies[4 - n1 - 1, n2, 4 - n3 - 1] = b[(2 * n3) + 1].GetPixel((2 * n1) + 1, 2 * n2).ToArgb();
                        f.frequencies[4 - n1 - 1, 4 - n2 - 1, n3] = b[2 * n3].GetPixel((2 * n1) + 1, (2 * n2) + 1).ToArgb();
                        f.frequencies[4 - n1 - 1, 4 - n2 - 1, 4 - n3 - 1] = b[(2 * n3) + 1].GetPixel((2 * n1) + 1, (2 * n2) + 1).ToArgb();
                    }


            /* Butterfly */
            for (int stage = 1; stage <= m; stage++)
            {
                ns = ns >> 1;
                ns2 = ns >> 1;
                float r2_n = pi / (ns << 1);
                for (int i1 = 0; i1 < 4; i1 += ns)
                    for (int i2 = 0; i2 < 4; i2 += ns)
                        for (int i3 = 0; i3 < 4; i3 += ns)
                        {
                            for (int k1 = 0; k1 < ns2; k1++)
                            {
                                int s11 = k1 + i1;
                                int s13 = k1 + ns2;
                                float Cfac = (float)Math.Cos(r2_n * ((k1 << 2) + 1)) * 2;

                                for (int k2 = 0; k2 < ns2; k2++)
                                {
                                    int s21 = k2 + i2;
                                    int s23 = k2 + ns2;
                                    float Cfac1 = (float)Math.Cos(r2_n * ((k2 << 2) + 1)) * 2;

                                    for (int k3 = 0; k3 < ns2; k3++)
                                    {
                                        int s31 = k3 + i3;
                                        int s33 = k3 + ns2;
                                        float Cfac2 = (float)Math.Cos(r2_n * ((k3 << 2)) + 1) * 2;

                                        p0 = (float)(f.frequencies[s11, s21, s31] + f.frequencies[s11, s21, s33]);
                                        p1 = (float)(f.frequencies[s11, s21, s31] - f.frequencies[s11, s21, s33]);
                                        p2 = (float)(f.frequencies[s11, s23, s31] + f.frequencies[s11, s23, s33]);
                                        p3 = (float)(f.frequencies[s11, s23, s31] - f.frequencies[s11, s23, s33]);
                                        p4 = (float)(f.frequencies[s13, s21, s31] + f.frequencies[s13, s21, s33]);
                                        p5 = (float)(f.frequencies[s13, s21, s31] - f.frequencies[s13, s21, s31]);
                                        p6 = (float)(f.frequencies[s13, s23, s31] + f.frequencies[s13, s23, s31]);
                                        p7 = (float)(f.frequencies[s13, s23, s31] - f.frequencies[s13, s23, s33]);

                                        p8 = p0 + p2;
                                        p9 = p1 + p3;
                                        p10 = p0 - p2;
                                        p11 = p1 - p3;

                                        p0 = p4 + p6;
                                        p1 = p5 + p7;
                                        p2 = p4 - p6;
                                        p3 = p5 - p7;

                                        Cfacm = Cfac1 * Cfac2;

                                        f.frequencies[s11, s21, s31] = p8 + p0;
                                        f.frequencies[s13, s21, s31] = (p8 - p0) * Cfac;
                                        f.frequencies[s11, s23, s31] = (p10 + p2) * Cfac1;
                                        f.frequencies[s11, s21, s33] = (p9 + p1) * Cfac2;
                                        f.frequencies[s13, s23, s31] = (p10 - p2) * Cfac * Cfac1;
                                        f.frequencies[s13, s21, s33] = (p9 - p1) * Cfac * Cfac2;
                                        f.frequencies[s11, s23, s33] = (p11 + p3) * Cfacm;
                                        f.frequencies[s13, s23, s33] = (p11 - p3) * Cfac * Cfacm;


                                        //mult. by transform factor
                                        //f.frequencies[k1, k2, 0] = f.frequencies[k1, k2, 0] * p0;

                                    }
                                }
                            }
                        }
            }


            return f;
        }

        /// <summary>
        /// DCT 3D of video block
        /// </summary>
        /// <param name="b0"></param>
        /// <param name="b1"></param>
        /// <param name="b2"></param>
        /// <param name="b3"></param>
        /// <param name="mode">Division mode, to reduce computational complexity. 0 for high frequencies 1 for low</param>
        /// <returns></returns>
        static Frequency_4x4x4 DCT3D_4x4x4(Bitmap b0, Bitmap b1, Bitmap b2, Bitmap b3, int mode)
        {

            Bitmap[] b = { b0, b1, b2, b3 };

            int beta_1 = (mode == 0) ? 2 : 0;
            int beta_2 = (mode == 0) ? 4 : 2;

            Frequency_4x4x4 f = new Frequency_4x4x4();
            int x = 0;
            int y = 0;
            int i = 0;
            int j = 0;
            int k = 0;
            int z = 0;

            double summand = 0;

            for (z = 0; z < 4; z++)
            {
                for (y = 0; y < 4; y++)
                {
                    for (x = 0; x < 4; x++)
                    {
                        for (k = 0; k < 4; k++)
                        {
                            for (i = 0; i < 4; i++)
                            {
                                for (j = 0; j < 4; j++)
                                {
                                    summand += (b[k].GetPixel(j, i).ToArgb() * Math.Cos(Math.PI * z * ((2 * k) + 1) / 8) * Math.Cos(Math.PI * y * ((2 * i) + 1) / 8) * Math.Cos(Math.PI * x * ((2 * j) + 1) / 8));
                                }
                            }
                        }
                        //summand = 0.5 * summand;
                        if (x == 0) summand *= (0.5);
                        if (y == 0) summand *= (0.5);
                        if (z == 0) summand *= (0.5);
                        if (x > 0) summand *= (1.0 / Math.Sqrt(2));
                        if (y > 0) summand *= (1.0 / Math.Sqrt(2));
                        if (z > 0) summand *= (1.0 / Math.Sqrt(2));
                        f.frequencies[x, y, z] = summand;
                        summand = 0;
                    }
                }
            }
            return f;
        }

        /// <summary>
        /// RC Decomposition DCT 3D of video block
        /// </summary>
        /// <param name="b0"></param>
        /// <param name="b1"></param>
        /// <param name="b2"></param>
        /// <param name="b3"></param>
        /// <param name="mode">Division mode, to reduce computational complexity. 0 for high frequencies 1 for low</param>
        /// <returns></returns>
        static Frequency_4x4x4 Faster_DCT3D_4x4x4(Bitmap b0, Bitmap b1, Bitmap b2, Bitmap b3, int mode)
        {

            Bitmap[] b = { b0, b1, b2, b3 };


            Frequency_4x4x4 f = new Frequency_4x4x4();
            Frequency_4x4x4 f2 = new Frequency_4x4x4();
            Frequency_4x4x4 f3 = new Frequency_4x4x4();

            int x = 0;
            int y = 0;
            int j = 0;
            int z = 0;

            double summand = 0;
            
            for (z = 0; z < 4; z++)
                {
                for (y = 0; y < 4; y++)
                    {
                    for (x = 0; x < 4; x++)
                        {
                                for (j = 0; j < 4; j++)
                                {
                                    summand += (b[z].GetPixel(j, y).ToArgb() * Math.Cos(Math.PI * x * ((2 * j) + 1) / 8));
                                }

                                //summand = 0.5 * summand;
                            if (x == 0) summand *= (0.5);
                            if (x > 0) summand *= (1.0 / Math.Sqrt(2));
                            f.frequencies[x, y, z] = summand;
                            summand = 0;
                        }
                    }
                }
            for (z = 0; z < 4; z++)
            {
                for (y = 0; y < 4; y++)
                {
                    for (x = 0; x < 4; x++)
                    {
                        for (j = 0; j < 4; j++)
                        {
                            summand += (f.frequencies[y, j, z] * Math.Cos(Math.PI * x * ((2 * j) + 1) / 8));
                        }
                        //summand = 0.5 * summand;
                        if (x == 0) summand *= (0.5);
                        if (x > 0) summand *= (1.0 / Math.Sqrt(2));
                        f2.frequencies[y, x, z] = summand;
                        summand = 0;
                    }
                }
            }

            for (z = 0; z < 4; z++)
            {
                for (y = 0; y < 4; y++)
                {
                    for (x = 0; x < 4; x++)
                    {
                        for (j = 0; j < 4; j++)
                        {
                            summand += (f2.frequencies[z, y, j] * Math.Cos(Math.PI * x * ((2 * j) + 1) / 8));
                        }
                        //summand = 0.5 * summand;
                        if (x == 0) summand *= (0.5);
                        if (x > 0) summand *= (1.0 / Math.Sqrt(2));
                        f3.frequencies[z, y, x] = summand;
                        summand = 0;
                    }
                }
            }
            return f3;
        }

        /// <summary>
        /// Daubechies Wavelet Transform of Bitmap
        /// </summary>
        /// <param name="b"></param>
        /// <returns></returns>
    static double[,] Wavelet_Daub(Bitmap b)
    {
            double[,] a = new double[b.Width, b.Height];
            int i, j;
            
            double[,] ab = new double[b.Width, b.Height];

            for (i = 0; i < b.Height; i++)
                for (j = 0; j < b.Width; j++)
                    a[j, i] = b.GetPixel(j, i).ToArgb();
            

      double h0 = (1 + Math.Sqrt(3))/(4*Math.Sqrt(2));
      double h1 = (3 + Math.Sqrt(3))/(4*Math.Sqrt(2));
      double h2 = (3 - Math.Sqrt(3))/(4*Math.Sqrt(2));
      double h3 = (1 - Math.Sqrt(3))/(4*Math.Sqrt(2));
      
      double g0 =  h3;
      double g1 = -h2;
      double g2 =  h1;
      double g3 = -h0;

      
         int n = b.Width;
         int half = n/2;
         
         
         double[] tmp = new double[n];
         
    for (int k = 0; k < b.Height; k++)
    {
        i = 0;
        j = 0;

         for (j = 0; j < n-3; j = j + 2) {
            tmp[i]      = a[j, k]*h0 + a[j+1, k]*h1 + a[j+2, k]*h2 + a[j+3, k]*h3;
            tmp[i+half] = a[j, k]*g0 + a[j+1, k]*g1 + a[j+2, k]*g2 + a[j+3, k]*g3;
            i++;
         }

         tmp[i]      = a[n-2, k]*h0 + a[n-1, k]*h1 + a[0, k]*h2 + a[1, k]*h3;
         tmp[i+half] = a[n-2, k]*g0 + a[n-1, k]*g1 + a[0, k]*g2 + a[1, k]*g3;

         for (i = 0; i < n; i++) {
            a[i, k] = tmp[i];
         }
    }

    n = b.Height;
    half = n / 2;

    tmp = new double[n];
        
        for (int k = 0; k < b.Width; k++)
    {
        i = 0;
        j = 0;

         for (j = 0; j < n-3; j = j + 2) {
            tmp[i]      = a[k, j]*h0 + a[k, j+1]*h1 + a[k, j+2]*h2 + a[k, j+3]*h3;
            tmp[i+half] = a[k, j]*g0 + a[k, j+1]*g1 + a[k, j+2]*g2 + a[k, j+3]*g3;
            i++;
         }

         tmp[i]      = a[k, n-2]*h0 + a[k, n-1]*h1 + a[k, 0]*h2 + a[k, 1]*h3;
         tmp[i+half] = a[k, n-2]*g0 + a[k, n-1]*g1 + a[k, 0]*g2 + a[k, 1]*g3;

         for (i = 0; i < n; i++) {
            a[k, i] = tmp[i];
         }
    }


        for (int y = 0; y < b.Height; y++)
            for (int x = 0; x < b.Width; x++)
                 ab[x, y] = a[x, y];
        return ab;
        }
    

    /// <summary>
    /// Inverse Daubechies Wavelet Transform of Bitmap
    /// </summary>
    /// <param name="b"></param>
    /// <returns></returns>
    static Bitmap iWavelet_Daub(double[,] b)
    {
        double[,] a = new double[b.GetLength(0), b.GetLength(1)];
        int i, j;

        Bitmap ab = new Bitmap(b.GetLength(0), b.GetLength(1));

        for (i = 0; i < b.GetLength(1); i++)
            for (j = 0; j < b.GetLength(0); j++)
                a[j, i] = b[j, i];


        double Ih0 = (3 - Math.Sqrt(3)) / (4 * Math.Sqrt(2));
        double Ih1 = (3 + Math.Sqrt(3)) / (4 * Math.Sqrt(2));
        double Ih2 = (1 + Math.Sqrt(3)) / (4 * Math.Sqrt(2));
        double Ih3 = (1 - Math.Sqrt(3)) / (4 * Math.Sqrt(2));

        double Ig0 = (1 - Math.Sqrt(3)) / (4 * Math.Sqrt(2));
        double Ig1 = -(1 + Math.Sqrt(3)) / (4 * Math.Sqrt(2)); ;
        double Ig2 = (3 + Math.Sqrt(3)) / (4 * Math.Sqrt(2));
        double Ig3 = -(3 - Math.Sqrt(3)) / (4 * Math.Sqrt(2));

        int n = b.GetLength(0);
        int half = n / 2;
        int halfPls1 = half + 1;


        double[] tmp = new double[n];

        for (int k = 0; k < b.GetLength(1); k++)
        {
            i = 0;
            j = 0;

            tmp[0] = a[half - 1, k] * Ih0 + a[n - 1, k] * Ih1 + a[0, k] * Ih2 + a[half, k] * Ih3;
            tmp[1] = a[half - 1, k] * Ig0 + a[n - 1, k] * Ig1 + a[0, k] * Ig2 + a[half, k] * Ig3;
            j = 2;
            for (i = 0; i < half - 1; i++)
            {
                //     smooth val     coef. val       smooth val    coef. val
                tmp[j++] = a[i, k] * Ih0 + a[i + half, k] * Ih1 + a[i + 1, k] * Ih2 + a[i + halfPls1, k] * Ih3;
                tmp[j++] = a[i, k] * Ig0 + a[i + half, k] * Ig1 + a[i + 1, k] * Ig2 + a[i + halfPls1, k] * Ig3;
            }
            for (i = 0; i < n; i++)
            {
                a[i, k] = tmp[i];
            }
        }

        n = b.GetLength(1);
        half = n / 2;
        halfPls1 = half + 1;


        tmp = new double[n];

        for (int k = 0; k < b.GetLength(0); k++)
        {
            i = 0;
            j = 0;

            tmp[0] = a[k, half - 1] * Ih0 + a[k, n - 1] * Ih1 + a[k, 0] * Ih2 + a[k, half] * Ih3;
            tmp[1] = a[k, half - 1] * Ig0 + a[k, n - 1] * Ig1 + a[k, 0] * Ig2 + a[k, half] * Ig3;
            j = 2;
            for (i = 0; i < half - 1; i++)
            {
                //     smooth val     coef. val       smooth val    coef. val
                tmp[j++] = a[k, i] * Ih0 + a[k, i + half] * Ih1 + a[k, i + 1] * Ih2 + a[k, i + halfPls1] * Ih3;
                tmp[j++] = a[k, i] * Ig0 + a[k, i + half] * Ig1 + a[k, i + 1] * Ig2 + a[k, i + halfPls1] * Ig3;
            }
            for (i = 0; i < n; i++)
            {
                a[k, i] = tmp[i];
            }
        }


        for (int y = 0; y < b.GetLength(1); y++)
            for (int x = 0; x < b.GetLength(0); x++)
                ab.SetPixel(x, y, Color.FromArgb((int)Math.Round(a[x, y])));
        return ab;
    }

        static Frequency_4x4 DCT2D_4x4(Bitmap b)
        {

            Frequency_4x4 f = new Frequency_4x4();
            int x = 0;
            int y = 0;
            int i = 0;
            int j = 0;

            double summand = 0;

            for (y = 0; y < 4; y++)
            {
                for (x = 0; x < 4; x++)
                {
                    for (i = 0; i < 4; i++)
                    {
                        for (j = 0; j < 4; j++)
                        {
                            summand += (b.GetPixel(j, i).ToArgb() * Math.Cos(Math.PI * y * ((2 * i) + 1) / 8) * Math.Cos(Math.PI * x * ((2 * j) + 1) / 8));
                        }
                    }
                    //summand = 0.5 * summand;
                    if (x == 0) summand *= (0.5);
                    if (y == 0) summand *= (0.5);
                    if (x > 0) summand *= (1.0 / Math.Sqrt(2));
                    if (y > 0) summand *= (1.0 / Math.Sqrt(2));
                    f.frequencies[x, y] = summand;
                    summand = 0;
                }
            }
            return f;
        }

        static Bitmap iDCT2D_4x4(Frequency_4x4 f)
        {
            Bitmap b = new Bitmap(4, 4);
            int x = 0;
            int y = 0;
            int i = 0;
            int j = 0;

            double summand = 0;
            double temp = 0;
            for (y = 0; y < 4; y++)
            {
                for (x = 0; x < 4; x++)
                {
                    for (i = 0; i < 4; i++)
                    {
                        for (j = 0; j < 4; j++)
                        {
                            temp = f.frequencies[j, i] * Math.Cos(Math.PI * i * ((2 * y) + 1) / 8) * Math.Cos(Math.PI * j * ((2 * x) + 1) / 8);
                            if (i == 0) temp *= (0.5);
                            if (j == 0) temp *= (0.5);
                            if (i > 0) temp *= (1.0 / Math.Sqrt(2));
                            if (j > 0) temp *= (1.0 / Math.Sqrt(2));
                            //summand *= 0.5;
                            summand += temp;
                        }
                    }
                    //summand = 0.5 * summand;

                    b.SetPixel(x, y, Color.FromArgb((int)Math.Round(summand)));
                    summand = 0;
                }
            }
            return b;
        }


        static Bitmap[] iDCT3D_4x4x4(Frequency_4x4x4 f)
        {
            Bitmap[] b = new Bitmap[4];
            b[0] = new Bitmap(4, 4);
            b[1] = new Bitmap(4, 4);
            b[2] = new Bitmap(4, 4);
            b[3] = new Bitmap(4, 4);

            int x = 0;
            int y = 0;
            int z = 0;
            int i = 0;
            int j = 0;
            int k = 0;


            double summand = 0;
            double temp = 0;
            for (z = 0; z < 4; z++)
            {
                for (y = 0; y < 4; y++)
                {
                    for (x = 0; x < 4; x++)
                    {
                        for (k = 0; k < 4; k++)
                        {
                            for (i = 0; i < 4; i++)
                            {
                                for (j = 0; j < 4; j++)
                                {
                                    temp = f.frequencies[j, i, k] * Math.Cos(Math.PI * k * ((2 * z) + 1) / 8) * Math.Cos(Math.PI * i * ((2 * y) + 1) / 8) * Math.Cos(Math.PI * j * ((2 * x) + 1) / 8);
                                    if (i == 0) temp *= (0.5);
                                    if (j == 0) temp *= (0.5);
                                    if (i > 0) temp *= (1.0 / Math.Sqrt(2));
                                    if (j > 0) temp *= (1.0 / Math.Sqrt(2));
                                    //summand *= 0.5;
                                    summand += temp;
                                }
                            }
                        }
                        //summand = 0.5 * summand;

                        b[z].SetPixel(x, y, Color.FromArgb((int)Math.Round(summand)));
                        summand = 0;
                    }
                }
            }
            return b;
        }


        /// <summary>
        /// Embed the bit within the 8 dimensional vector
        /// </summary>
        /// <param name="f"></param>
        /// <returns></returns>
        static Frequency_4x4 vector_Embed8D(Frequency_4x4 f, int T, byte bit)
        {
            double distance = Math.Pow(f.frequencies[0, 0], 2) + Math.Pow(f.frequencies[1, 0], 2) + Math.Pow(f.frequencies[0, 1], 2) + Math.Pow(f.frequencies[0, 2], 2) + Math.Pow(f.frequencies[1, 1], 2) + Math.Pow(f.frequencies[2, 0], 2) + Math.Pow(f.frequencies[3, 0], 2) + Math.Pow(f.frequencies[2, 1], 2);
            distance = Math.Sqrt(distance);
            double distance_prime = Math.Round(distance / (double)T);
            if (bit > 0)
                distance_prime += 0.25;
            else
                distance_prime -= 0.25;
            distance_prime = distance_prime * (double)T;
            double coeff = distance_prime / distance;
            f.frequencies[0, 0] *= coeff;
            f.frequencies[1, 0] *= coeff;
            f.frequencies[0, 1] *= coeff;
            f.frequencies[0, 2] *= coeff;
            f.frequencies[1, 1] *= coeff;
            f.frequencies[2, 0] *= coeff;
            f.frequencies[3, 0] *= coeff;
            f.frequencies[2, 1] *= coeff;

            return f;
        }


        /// <summary>
        /// Retrieves the bit from within the 8 dimensional vector
        /// </summary>
        /// <param name="f"></param>
        /// <returns></returns>
        static byte vector_Retrieve8D(Frequency_4x4 f, int T)
        {
            byte bit = 0;
            double distance = Math.Pow(f.frequencies[0, 0], 2) + Math.Pow(f.frequencies[1, 0], 2) + Math.Pow(f.frequencies[0, 1], 2) + Math.Pow(f.frequencies[0, 2], 2) + Math.Pow(f.frequencies[1, 1], 2) + Math.Pow(f.frequencies[2, 0], 2) + Math.Pow(f.frequencies[3, 0], 2) + Math.Pow(f.frequencies[2, 1], 2);
            distance = Math.Sqrt(distance);
            double distance_prime = distance / (double)T;
            distance_prime = distance_prime - Math.Round(distance_prime);
            if (distance_prime >= 0)
                bit = 1;
            else
                bit = 0;

            return bit;
        }

        /// <summary>
        /// Calculates the alternating sign matrix for a spatial domain block
        /// </summary>
        /// <param name="block">The block of pixel values</param>
        /// <returns></returns>
        static int[,] matrix_SGN(double[,] block)
        {
            int N = block.GetLength(0);
            int M = block.GetLength(1);

            int[,] sgn = new int[N, M];
            for (int i = 0; i < M; i++)
                for (int j = 0; j < N; j++)
                    sgn[j, i] = ((((i) % 2) == ((j) % 2)) ? 1 : -1);
            return sgn;
        }

        /// <summary>
        /// Calculates the arithmetic difference \alpha of the block
        /// </summary>
        /// <param name="block"></param>
        static int arithmetic_Difference(double[,] block, int[,] sgn)
        {
            int N = block.GetLength(0);
            int M = block.GetLength(1);

            int alpha = 0;
            for (int i = 0; i < M; i++)
                for (int j = 0; j < N; j++)
                    alpha += (sgn[j, i] * (int)block[j, i]);
            return alpha;
        }

        /// <summary>
        /// Calculates the arithmetic threshold of the block
        /// </summary>
        /// <param name="block">The Block</param>
        /// <param name="alpha">The Arithmetic Difference</param>
        /// <param name="gamma">Gamma Value</param>
        /// <returns></returns>
        static double[,] arithmetic_Threshold(double[,] block, int alpha, double gamma, int T)
        {
            int N = block.GetLength(0);
            int M = block.GetLength(1);

            for (int i = 0; i < M; i++)
                for (int j = 0; j < N; j++)
                {
                    if ((((i + 1) % 2) == ((j + 1) % 2)) && alpha > T) block[j, i] += gamma;
                    if ((((i + 1) % 2) != ((j + 1) % 2)) && alpha < -T) block[j, i] += gamma;
                }
            return block;
        }

        /// <summary>
        /// Embeds a bit within the current block. Returns a block to be stored.
        /// </summary>
        /// <param name="bit"></param>
        /// <param name="alpha"></param>
        /// <param name="block"></param>
        /// <param name="gamma_prime"></param>
        /// <param name="T"></param>
        /// <returns></returns>
        static double[,] arithmetic_Embed(byte bit, int alpha, double[,] block, double gamma_prime, int T)
        {
            int N = block.GetLength(0);
            int M = block.GetLength(1);

            if (bit > 0)
            {
                for (int i = 0; i < M; i++)
                    for (int j = 0; j < N; j++)
                    {
                        if (((((i + 1) % 2) == ((j + 1) % 2)) && alpha >= 0 && alpha <= T)) block[j, i] += gamma_prime;
                        if (((((i + 1) % 2) != ((j + 1) % 2)) && alpha < 0 && alpha >= -T)) block[j, i] += gamma_prime;
                    }
            }
            return block;
        }

        /// <summary>
        /// Retrieves a bit from the current block
        /// </summary>
        /// <param name="block"></param>
        /// <param name="T"></param>
        /// <param name="G"></param>
        /// <returns></returns>
        static byte arithmetic_Retrieve(double[,] block, int T, int G)
        {
            int N = block.GetLength(0);
            int M = block.GetLength(1);

            int alpha = arithmetic_Difference(block, matrix_SGN(block));

            for (int i = 0; i < M; i++)
                for (int j = 0; j < N; j++)
                {
                    if (alpha >= -T && alpha <= T) return 0;
                    if ((alpha > T && alpha <= (2 * T) + G) || (alpha < -T && alpha >= -((2 * T) + G))) return 1;
                }
            Console.Write("?"); //salt
            return 1;
        }

        static double[,] Array2DFromBitmap(Bitmap b)
        {
            double[,] a = new double[b.Width, b.Height];
            for (int i = 0; i < b.Height; i++)
                for (int j = 0; j < b.Width; j++)
                    a[j, i] = (b.GetPixel(j, i).ToArgb());
            return a;
        }

        static Bitmap BitmapFromArray2D(double[,] a)
        {
            Bitmap b = new Bitmap(a.GetLength(0), a.GetLength(1));
            for (int i = 0; i < b.Height; i++)
                for (int j = 0; j < b.Width; j++)
                    b.SetPixel(j, i, Color.FromArgb((int)a[j, i]));
            return b;
        }

        static byte[] Array1DFromBitmap(Bitmap b, int size)
        {
            byte[] a = new byte[size];
            BitmapData data = b.LockBits(new Rectangle(0, 0, b.Width, b.Height), ImageLockMode.ReadWrite, PixelFormat.Format24bppRgb);
            int stride = data.Stride;
            unsafe
            {
                byte* ptr = (byte*)data.Scan0;
                for (int i = 0; i < b.Height; i++)
                    for (int j = 0; j < b.Width; j++)
                    {
                        //a[(b.Width * i) + j] = (byte)(b.GetPixel(j, i).ToArgb());
                        // layer.GetBitmap().SetPixel(x, y, m_colour);
                        a[(b.Width * i) + j] = ptr[(j * 3) + i * stride];
                        //ptr[(j * 3) + i * stride + 1] = m_colour.G;
                        //ptr[(j * 3) + i * stride + 2] = m_colour.R;
                    }
            }
            b.UnlockBits(data);
            return a;
        }

        static byte[] Array1DFromBitmapSafe(Bitmap b, int size)
        {
            byte[] a = new byte[size];
            //BitmapData data = b.LockBits(new Rectangle(0, 0, b.Width, b.Height), ImageLockMode.ReadWrite, PixelFormat.Format24bppRgb);
            //int stride = data.Stride;
            //unsafe
            //{
            //byte* ptr = (byte*)data.Scan0;    
            for (int i = 0; i < b.Height; i++)
                for (int j = 0; j < b.Width; j++)
                {
                    a[(b.Width * i) + j] = (byte)(b.GetPixel(j, i).ToArgb());
                    // layer.GetBitmap().SetPixel(x, y, m_colour);
                    //a[(b.Width * i) + j] = ptr[(j * 3) + i * stride];
                    //ptr[(j * 3) + i * stride + 1] = m_colour.G;
                    //ptr[(j * 3) + i * stride + 2] = m_colour.R;
                }
            //}
            //b.UnlockBits(data);
            return a;
        }

        static Bitmap BitmapFromArray1DSafe(byte[] a, int width, int height)
        {
            Bitmap b = new Bitmap(width, height);
            //BitmapData data = b.LockBits(new Rectangle(0, 0, b.Width, b.Height), ImageLockMode.ReadWrite, PixelFormat.Format24bppRgb);
            //int stride = data.Stride;
            //unsafe
            //{
            //byte* ptr = (byte*)data.Scan0;
            for (int i = 0; i < a.Length; i++)
                if (i / width < 720)
                    //for (int j = 0; j < b.Width; j++)
                    //{
                    //a[(b.Width * i) + j] = (byte)(b.GetPixel(j, i).ToArgb());
                    // layer.GetBitmap().SetPixel(x, y, m_colour);
                    //a[(b.Width * i % width) + i / width] = ptr[(j * 3) + i * stride];
                    //ptr[(i % width * 3) + ((i / width) * stride)] = a[i];
                    //ptr[(j * 3) + i * stride + 1] = m_colour.G;
                    //ptr[(j * 3) + i * stride + 2] = m_colour.R;
                    b.SetPixel(i % width, i / width, Color.FromArgb((int)a[i]));
            //}
            //
            //}
            //b.UnlockBits(data);
            return b;
        }

        /// <summary>
        /// Gives a running sum of an array
        /// </summary>
        /// <param name="input"></param>
        /// <param name="frames"></param>
        /// <returns></returns>
        static double[] AccArray(byte[] input, double[] prev)
        {
            byte[] result = new byte[input.Length * 8];

            for (int i = 0; i < input.Length; i++)
                for (int k = 0; k < 8; k++)
                {
                    result[(8 * i) + k] = (byte)(((input[i] >> k) % 2));
                    prev[(8 * i) + k] += result[(8 * i) + k];
                }
            // each slot of result contains either a zero or one
            return prev;
        }

        /// <summary>
        /// Gives a running average of an array
        /// </summary>
        /// <param name="input"></param>
        /// <param name="frames"></param>
        /// <returns></returns>
        static double[] AvgArray(double[] input, double frames)
        {
            double[] result = new double[input.Length];

            for (int i = 0; i < input.Length; i++)
                result[i] = input[i] / frames;

            // each slot of result contains either a zero or one
            return result;
        }

        /// <summary>
        /// Calculates the P value
        /// </summary>
        /// <param name="input"></param>
        /// <param name="frames"></param>
        /// <returns></returns>
        static double P(double[] input)
        {
            double result = 0;

            for (int i = 0; i < input.Length; i++)
                result += input[i];

            result /= input.Length;
            return Math.Pow(((2 * result) - 1), 2);
        }



        enum frequencyband {LL, HL, LH, HH};
        /// <summary>
        /// Returns the chosen subband of the image
        /// </summary>
        /// <param name="coefficients">A collection of DWT coefficients</param>
        /// <param name="f">The subband, LL, HL, LH, HH</param>
        /// <returns></returns>
        static Bitmap DWTtoBitmap(double[,] coefficients, frequencyband f)
        {
            Bitmap b = new Bitmap
                (
                coefficients.GetLength(0)/2, 
                coefficients.GetLength(1)/2
                );

            int k = ((int)f%2);
            int c = ((int)f/2);
            int kk = k*(coefficients.GetLength(0));
            int cc = c*(coefficients.GetLength(1));
            for (int i = kk/2; i < kk; i++)
                for (int j = cc/2; j < cc; j++)
                {
                    b.SetPixel(j, i, Color.FromArgb((int)Math.Round(coefficients[j, i])));
                }
            return b;
        }



        static Bitmap BitmapFromArray1D(byte[] a, int width, int height)
        {
            Bitmap b = new Bitmap(width, height);
            BitmapData data = b.LockBits(new Rectangle(0, 0, b.Width, b.Height), ImageLockMode.ReadWrite, PixelFormat.Format24bppRgb);
            int stride = data.Stride;
            unsafe
            {
                byte* ptr = (byte*)data.Scan0;
                for (int i = 0; i < a.Length; i++)
                    if (i / width < 720)
                        //for (int j = 0; j < b.Width; j++)
                        //{
                        //a[(b.Width * i) + j] = (byte)(b.GetPixel(j, i).ToArgb());
                        // layer.GetBitmap().SetPixel(x, y, m_colour);
                        //a[(b.Width * i % width) + i / width] = ptr[(j * 3) + i * stride];
                        ptr[(i % width * 3) + ((i / width) * stride)] = a[i];
                //ptr[(j * 3) + i * stride + 1] = m_colour.G;
                //ptr[(j * 3) + i * stride + 2] = m_colour.R;
                //}
                //b.SetPixel(i%width, i/width, Color.FromArgb((int)a[i]));
            }
            b.UnlockBits(data);
            return b;
        }

        /// <summary>
        /// Embeds a byte[] within the spatial domain of the video and returns the bitmap.
        /// </summary>
        /// <param name="b"></param>
        /// <returns></returns>
        static Bitmap spatial_Embed(Bitmap b, int block_size, int T, int G, byte[] bytes, int rep, int trep)
        {
            double[,] block = new double[block_size, block_size];
            double gamma = Gamma(block, T, G);
            double gamma_prime = Gamma_Prime(block, T, G);
            int[,] sgn = matrix_SGN(block);

            long temppos = pos;
            long temppos_rep = pos_rep;
            long temppos_bit = pos_bit;

            if (loop) pos = 0;

            if (loop) pos_bit = 0;

            if (loop) pos_rep = 0;

            Bitmap src = new Bitmap(block_size, block_size);
            int x = 0;
            int y = 0;

            for (; ; pos++)
            {

                if (pos == bytes.Length && loop == true) return b;
                if (pos == bytes.Length) pos = 0;




                for (pos_bit = 0; pos_bit < 8; pos_bit++)
                {
                    //int y = (block_size * (((8 *  i) + (  k))) / b.Width );
                    //int x = (block_size * (((8 *  i) + (  k))) % b.Width );




                    for (pos_rep = 0; pos_rep < rep; pos_rep++)
                    {
                        src = b.Clone(new Rectangle(x, y, block_size, block_size), b.PixelFormat);
                        block = Array2DFromBitmap(src);
                        byte bit = (byte)((bytes[pos] >> (byte)pos_bit) % 2);
                        int alpha = arithmetic_Difference(block, sgn);
                        block = arithmetic_Threshold(block, alpha, gamma, T);

                        alpha = arithmetic_Difference(block, sgn);//See below

                        if (alpha >= -T && alpha <= T) //Thank you Alavianamehr! for leaving important details out after plagiari- "paraphrasing" Xian
                            block = arithmetic_Embed(bit, alpha, block, gamma_prime, T);
                        if (alpha > T && alpha < -T)
                            pos_bit--;

                        src = BitmapFromArray2D(block);

                        //g.DrawImage(src, x, y, new Rectangle(0, 0, block_size, block_size), GraphicsUnit.Pixel);
                        for (int m = 0; m < block_size; m++)
                            for (int n = 0; n < block_size; n++)
                                b.SetPixel(x + n, y + m, src.GetPixel(n, m));
                        //src.Dispose();

                        x += block_size;
                        if (x >= b.Width) { y += block_size; x = 0; }


                        if (y >= b.Height)
                        {
                            tmod++;
                            if (!loop)
                            {
                                if (tmod % (trep + 1) != 0)
                                {
                                    //reset the values
                                    pos = temppos;
                                    pos_bit = temppos_bit;
                                    pos_rep = temppos_rep;
                                    tmod = 0;
                                }
                                else tmod = 0;
                            }
                            //otherwise, allow the parameters to advance

                            return b;
                        }
                    }
                    tmod++;



                }
            }
            //return b;
        }

        /// <summary>
        /// Embeds a byte[] within the transform domain of the video and returns the bitmap.
        /// </summary>
        /// <param name="b"></param>
        /// <returns></returns>
        static Bitmap transform_Embed(Bitmap b, int T, byte[] bytes, int rep)
        {

            int block_size = 4;
            double[,] block = new double[block_size, block_size];

            Bitmap src = new Bitmap(block_size, block_size);
            Frequency_4x4 f = new Frequency_4x4();
            int x = 0;
            int y = 0;

            if (loop) pos = 0;

            if (loop) pos_bit = 0;

            if (loop) pos_rep = 0;

            for (; ; pos++)
            {
                if (pos == bytes.Length && loop == true) return b;
                if (pos == bytes.Length) pos = 0;

                for (pos_bit = 0; pos_bit < 8; pos_bit++)
                {
                    for (pos_rep = 0; pos_rep < rep; pos_rep++)
                    {
                        src = b.Clone(new Rectangle(x, y, block_size, block_size), b.PixelFormat);
                        f = DCT2D_4x4(src);
                        byte bit = (byte)((bytes[pos] >> (byte)(pos_bit)) % 2);
                        f = vector_Embed8D(f, T, bit);

                        src = iDCT2D_4x4(f);
                        for (int m = 0; m < block_size; m++)
                            for (int n = 0; n < block_size; n++)
                                b.SetPixel(x + n, y + m, src.GetPixel(n, m));

                        //int foo = src.GetPixel(0, 0).ToArgb();
                        //foo = 0;

                        x += block_size;
                        if (x >= b.Width) { y += block_size; x = 0; }
                        if (y >= b.Height) return b;

                    }
                }
            }
            //return b;
        }


        /// <summary>
        /// DCT-DWT Algorithm. 
        /// IMPORTANT! THIS ONLY WORKS FOR 1080p video!
        /// 
        /// How this works:
        /// 
        /// Iteratively calculates the DWT of a video,
        /// until resolution is 144p and applies Yang's algorithm at each step.
        /// </summary>
        /// <param name="b"></param>
        /// <returns></returns>
        static Bitmap transform_Embed2(Bitmap b, int T, byte[] bytes, int rep)
        {

            int block_size = 4;
            double[,] block = new double[block_size, block_size];
            double[,] dwtcoeff;
            Bitmap src = new Bitmap(block_size, block_size);
            Frequency_4x4 f = new Frequency_4x4();
            
            if (b.Height != 1080)
            {
                Console.WriteLine("Error: video must be 1080p");
                while (true)
                {
                    Console.ReadLine();
                }
            }
            int x = 0;
            int y = 0;
            int kek = 1;
            
            Bitmap[] LL = new Bitmap[10]; //go magic numbers!
            LL[0] = b;

            //Continue wavelet decomposition as long as both dimensions 
            //of the video are still divisible by 2
            for (; b.Height/((int)Math.Pow(2, kek)) % 2 == 0
                && b.Width/((int)Math.Pow(2, kek)) % 2 == 0; kek++)
            {


            dwtcoeff = Wavelet_Daub(LL[kek-1]);
            LL[kek] = DWTtoBitmap(dwtcoeff, frequencyband.LL);

            if (loop) pos = 0;

            if (loop) pos_bit = 0;

            if (loop) pos_rep = 0;

            for (; ; pos++)
            {
                if (pos == bytes.Length && loop == true) return b;
                if (pos == bytes.Length) pos = 0;

                for (pos_bit = 0; pos_bit < 8; pos_bit++)
                {
                    for (pos_rep = 0; pos_rep < rep; pos_rep++)
                    {
                        src = LL[kek].Clone(new Rectangle(x, y, block_size, block_size), LL[kek].PixelFormat);
                        f = DCT2D_4x4(src);
                        byte bit = (byte)((bytes[pos] >> (byte)(pos_bit)) % 2);
                        
                        src = iDCT2D_4x4(f);
                        for (int m = 0; m < block_size; m++)
                            for (int n = 0; n < block_size; n++)
                                LL[kek].SetPixel(x + n, y + m, src.GetPixel(n, m));

                        //int foo = src.GetPixel(0, 0).ToArgb();
                        //foo = 0;

                        x += block_size;
                        /*
                        if (x >= LL[kek].Width) { y += block_size; x = 0; }
                        if (y >= LL[kek].Height) return b;
                        */

                    }
                }
            }
            //return b;
            double 
            for (int iii = 0; iii < kek; iii++)
            iWavelet_Daub(
            }
            
        }


        /// <summary>
        /// Embeds a byte[] within the transform domain of the video and returns the bitmap.
        /// </summary>
        /// <param name="b"></param>
        /// <returns></returns>
        static Bitmap barrage_Noise(Bitmap b, int T, byte[] bytes, int rep)
        {

            int block_size = 4;
            double[,] block = new double[block_size, block_size];

            Bitmap src = new Bitmap(block_size, block_size);
            Frequency_4x4 f = new Frequency_4x4();
            int x = 0;
            int y = 0;

            if (loop) pos = 0;

            if (loop) pos_bit = 0;

            if (loop) pos_rep = 0;

            for (; ; pos++)
            {
                if (pos == bytes.Length && loop == true) return b;
                if (pos == bytes.Length) pos = 0;

                for (pos_bit = 0; pos_bit < 8; pos_bit++)
                {
                    for (pos_rep = 0; pos_rep < rep; pos_rep++)
                    {
                        src = b.Clone(new Rectangle(x, y, block_size, block_size), b.PixelFormat);
                        f = DCT2D_4x4(src);
                        byte bit = (byte)((bytes[pos] >> (byte)(pos_bit)) % 2);
                        f = vector_Embed8D(f, 5, bit);

                        //Low Pass Filter :P
                        //src = iDCT2D_4x4(f);
                        f.frequencies[0, 3] = 0;
                        f.frequencies[1, 3] = 0;
                        f.frequencies[2, 3] = 0;
                        f.frequencies[3, 3] = 0;
                        f.frequencies[3, 2] = 0;
                        f.frequencies[3, 1] = 0;
                        f.frequencies[3, 0] = 0;
                        f.frequencies[2, 2] = 0;

                        src = iDCT2D_4x4(f);

                        for (int m = 0; m < block_size; m++)
                            for (int n = 0; n < block_size; n++)
                                b.SetPixel(x + n, y + m, src.GetPixel(n, m));

                        //int foo = src.GetPixel(0, 0).ToArgb();
                        //foo = 0;

                        x += block_size;
                        if (x >= b.Width) { y += block_size; x = 0; }
                        if (y >= b.Height) return b;

                    }
                }
            }
            //return b;
        }

        /// <summary>
        /// Retrieves a byte[] within the spatial domain of the video and returns the byte[].
        /// </summary>
        /// <param name="b"></param>
        /// <returns></returns>
        static byte[] spatial_Retrieve(Bitmap b, int block_size, int T, int G, byte[] bytes, int rep, int trep)
        {
            double[,] block = new double[block_size, block_size];
            double gamma = Gamma(block, T, G);
            double gamma_prime = Gamma_Prime(block, T, G);

            Bitmap src = new Bitmap(block_size, block_size);
            int x = 0;
            int y = 0;
            byte bit = 0;
            byte avg = 0;

            long temppos = pos;
            long temppos_rep = pos_rep;
            long temppos_bit = pos_bit;


            if (loop) pos = 0;


            if (loop) pos_bit = 0;

            if (loop) pos_rep = 0;

            byte[] newbytes = new byte[bytes.Length];

            for (; ; pos++)
            {
                if (pos > bytes.Length && loop == true) return newbytes;
                if (pos > bytes.Length && loop == false) pos = 0;


                for (pos_bit = 0; pos_bit < 8; pos_bit++)
                {
                    for (pos_rep = 0; pos_rep < rep; pos_rep++)
                    {
                        src = b.Clone(new Rectangle(x, y, block_size, block_size), b.PixelFormat);
                        block = Array2DFromBitmap(src);

                        int alpha = arithmetic_Difference(block, matrix_SGN(block));
                        //if (alpha <= ((2*T) + G) && alpha >= -((2*T)+ G)) //see my frustration above at Alavianmehr et. als' "original" work
                        //{
                        avg += arithmetic_Retrieve(block, T, G);


                        //}
                        //if (alpha > ((2 * T) + G) || alpha < -((2 * T) + G)) k--;

                        x += block_size;
                        if (x >= b.Width) { y += block_size; x = 0; }
                        if (y >= b.Height)
                        {
                            tmod++;
                            if (!loop)
                            {
                                if (tmod % trep != 0)
                                {
                                    //reset the values
                                    pos = temppos;
                                    pos_bit = temppos_bit;
                                    pos_rep = temppos_rep;
                                    tmod = 0;
                                }
                                else
                                    tmod = 0;
                            }
                            //otherwise, allow the parameters to advance
                            return newbytes;
                        }
                    }
                    bit = (byte)Math.Round((double)avg / (double)rep);
                    newbytes[pos] = (byte)(newbytes[pos] >> 1);
                    newbytes[pos] += (byte)(128 * bit);
                    avg = 0;
                }
            }
            //return newbytes;
        }



        /// <summary>
        /// Retrieves a byte[] within the transform domain of the video and returns the byte[].
        /// </summary>
        /// <param name="b"></param>
        /// <returns></returns>
        static byte[] transform_Retrieve(Bitmap b, byte[] bytes, int T, int rep)
        {
            double[,] block = new double[4, 4];

            if (loop) pos = 0;

            if (loop) pos_bit = 0;

            if (loop) pos_rep = 0;

            Bitmap src = new Bitmap(4, 4);
            int x = 0;
            int y = 0;
            byte bit = 0;
            byte avg = 0;
            Frequency_4x4 f = new Frequency_4x4();
            byte[] newbytes = new byte[bytes.Length];

            for (; ; pos++)
            {
                if (pos > bytes.Length && loop == true) return newbytes;
                if (pos > bytes.Length) pos = 0;

                for (pos_bit = 0; pos_bit < 8; pos_bit++)
                {
                    for (pos_rep = 0; pos_rep < rep; pos_rep++)
                    {
                        src = b.Clone(new Rectangle(x, y, 4, 4), b.PixelFormat);
                        f = DCT2D_4x4(src);
                        //block = Array2DFromBitmap(src);

                        //int alpha = arithmetic_Difference(block, matrix_SGN(block));
                        //if (alpha <= ((2*T) + G) && alpha >= -((2*T)+ G)) //see my frustration above at Alavianmehr et. als' "original" work
                        //{
                        avg += vector_Retrieve8D(f, T);


                        //}
                        //if (alpha > ((2 * T) + G) || alpha < -((2 * T) + G)) k--;

                        x += 4;
                        if (x >= b.Width) { y += 4; x = 0; }
                        if (y >= b.Height)
                            return newbytes;
                    }
                    bit = (byte)Math.Round((double)avg / (double)rep);
                    newbytes[pos] = (byte)(newbytes[pos] >> 1);
                    newbytes[pos] += (byte)(128 * bit);
                    avg = 0;
                }
            }
            //return newbytes;
        }


        static double MSE(Bitmap b1, Bitmap b2)
        {
            double mse = 0;
            for (int i = 0; i < b1.Height; i++)
                for (int j = 0; j < b1.Width; j++)
                    mse += Math.Pow(b1.GetPixel(j, i).ToArgb() - b2.GetPixel(j, i).ToArgb(), 2.0);
            mse /= (b1.Width * b1.Height);
            return mse;
        }

        static double PSNR(double MSE)
        {
            return 10 * Math.Log10(Math.Pow(256, 2) / MSE);
        }

        static Bitmap GetBMPFromFrame(string filename, int frame, int VideoWidth, int VideoHeight)
        {
            Bitmap b = new Bitmap(VideoWidth, VideoHeight);

            return b;
        }

        /// <summary>
        /// Calculates Gamma Value
        /// </summary>
        /// <param name="block">The block</param>
        /// <returns></returns>
        static double Gamma(double[,] block, int T, int G)
        {
            int N = block.GetLength(0);
            int M = block.GetLength(1);

            return (double)Math.Ceiling((2 * ((2 * G) + T)) / (double)(N * M));
        }

        /// <summary>
        /// Calculates Gamma_Prime Value
        /// </summary>
        /// <param name="block"></param>
        /// <param name="T"></param>
        /// <param name="G"></param>
        /// <returns></returns>
        static double Gamma_Prime(double[,] block, int T, int G)
        {
            int N = block.GetLength(0);
            int M = block.GetLength(1);


            return (double)Math.Ceiling((2 * (T + G)) / (double)(N * M));
        }

        /// <summary>
        /// Gets a byte[] from a file.
        /// </summary>
        /// <param name="fullFilePath"></param>
        /// <param name="offset"></param>
        /// <param name="length"></param>
        /// <returns></returns>
        public static byte[] GetBytesFromFile(string fullFilePath, int offset, int length)
        {
            FileStream fs = null;
            try
            {
                fs = File.OpenRead(fullFilePath);
                byte[] bytes = new byte[length];

                fs.Seek(offset, SeekOrigin.Begin);
                fs.Read(bytes, 0, length);
                return bytes;
            }
            finally
            {
                if (fs != null)
                {
                    fs.Close();
                    fs.Dispose();
                }
            }

        }

        /// <summary>
        /// Saves a byte[] to a file.
        /// </summary>
        /// <param name="fullFilePath"></param>
        /// <param name="offset"></param>
        /// <param name="length"></param>
        /// <returns></returns>
        public static byte[] SaveBytesToFile(string fullFilePath, int offset, int length, byte[] bytes)
        {
            FileStream fs = null;
            try
            {
                fs = File.OpenWrite(fullFilePath);
                //bytes = new byte[length];
                fs.Seek(offset, SeekOrigin.Begin);
                fs.Write(bytes, 0, length);

                return bytes;
            }
            finally
            {
                if (fs != null)
                {
                    fs.Close();
                    fs.Dispose();
                }
            }

        }



        /// <summary>
        /// Gets a byte[] from a file.
        /// </summary>
        /// <param name="fullFilePath"></param>
        /// <param name="offset"></param>
        /// <param name="length"></param>
        /// <returns></returns>
        public static byte[] GetBytesFromFile(string fullFilePath, int offset)
        {
            FileStream fs = null;
            try
            {
                fs = File.OpenRead(fullFilePath);
                byte[] bytes = new byte[fs.Length];

                fs.Seek(offset, SeekOrigin.Begin);
                fs.Read(bytes, 0, (int)bytes.Length);
                return bytes;
            }
            finally
            {
                if (fs != null)
                {
                    fs.Close();
                    fs.Dispose();
                }
            }

        }

        static double BER(byte[] a, byte[] b)
        {
            int i = 0;
            int k = 0;
            double ber = 0;



            for (i = 0; i < a.Length; i++)
                for (k = 0; k < 8; k++)
                    if (((a[i] >> k) % 2) != ((b[i] >> k) % 2))
                        ber++;


            return 100.0 * ber / a.Length / 8;

        }

        /// <summary>
        /// Takes the BER between a chunk of recovered data and the host data at a certain position and length
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        static double BER(byte[] src, byte[] rec, long position, int width)
        {
            int i = 0;
            int k = 0;
            double ber = 0;

            for (i = 0; i < width; i++)
                for (k = 0; k < 8; k++)
                    if (((src[i + position] >> k) % 2) != ((rec[i + position] >> k) % 2))
                        ber++;

            return 100.0 * ber / width / 8;

        }

        /// <summary>
        /// Expectation operator
        /// </summary>
        /// <param name="f"></param>
        /// <returns></returns>
        static double Expectation(double[] f)
        {
            return m(f);
        }
            /* wow this is embarrasing,  so much code for something entirely simple
            double sum = 0;
            int acc = 1;

            if (f.Length < 500) //O(n^2)
            {
                for (int i = 0; i < f.Length - 1; i++)
                {
                    for (int j = (i + 1); j < f.Length; j++)
                        if (f[i] == f[j]) acc++;
                    f[i] = acc * f[i];
                    acc = 1; // :-)
                }
                for (int i = 0; i < f.Length; i++)
                    sum += f[i];

                sum /= f.Length;
            }

            if (f.Length >= 500)  //O(nlogn)
            {
                double tweedledee, tweedledum = 0;
                
                tweedledee = f[0];
                Array.Sort(f); //O(nlogn)

                for (int i = 1; i < f.Length; i++) //O(n)
                {
                    tweedledum = f[i];
                    acc++;
                    if (tweedledum != tweedledee)
                    {
                        tweedledee = f[i];
                        acc = 0;
                        sum += (acc * tweedledum);
                    }
                }
                    
            }

            return sum;
             
        }
             * */

        /// <summary>
        /// Skewness
        /// </summary>
        /// <param name="mean"></param>
        /// <param name="std"></param>
        /// <param name="f"></param>
        /// <returns></returns>
        static double Skewness(double mean, double std, double[] f)
        {
            double[] h = new double[f.Length];
            for (int i = 0; i < f.Length; i++)
                h[i] = Math.Pow(((f[i] - mean) / std), 3);
            return Expectation(h);
        }

        /// <summary>
        /// Kurtosis
        /// </summary>
        /// <param name="mean"></param>
        /// <param name="std"></param>
        /// <param name="f"></param>
        /// <returns></returns>
        static double Kurtosis(double mean, double std, double[] f)
        {
            double[] h = new double[f.Length];
            for (int i = 0; i < f.Length; i++)
                h[i] = Math.Pow(((f[i] - mean) / std), 4);
            return Expectation(h);
        }

        /// <summary>
        /// Simply calculates the mean
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        static double m(double[] a)
        {
            double sum = 0; for (int i = 0; i < a.Length; i++)
                sum += a[i];
            sum /= a.Length; return sum;
        }

        /// <summary>
        /// Simply calculates the mean
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        static double m(double[, ,] a)
        {
            double sum = 0;
            for (int i = 0; i < a.GetLength(0); i++)
                for (int j = 0; j < a.GetLength(1); j++)
                    for (int k = 0; k < a.GetLength(2); k++)
                        sum += a[i, j, k];

            sum /= (a.GetLength(0) * a.GetLength(1) * a.GetLength(2)); return sum;
        }

        /// <summary>
        /// Simply calculates the population standard deviation
        /// </summary>
        /// <param name=f"a"></param>
        /// <returns></returns>
        static double std(double[, ,] a, double mean)
        {
            double sum = 0;
            for (int i = 0; i < a.GetLength(0); i++)
                for (int j = 0; j < a.GetLength(1); j++)
                    for (int k = 0; k < a.GetLength(2); k++)
                        sum += Math.Pow((a[i, j, k] - mean), 2);

            sum /= (a.GetLength(0) * a.GetLength(1) * a.GetLength(2));
            sum = Math.Sqrt(sum);

            return sum;
        }

        /// <summary>
        /// Flattens out a double[,,] to a double[]
        /// </summary>
        /// <returns></returns>
        static double[] flatten(double[, ,] d)
        {

            double[] foo = new double[d.GetLength(0) * d.GetLength(1) * d.GetLength(2)];

            for (int z = 0; z < d.GetLength(2); z++)
                for (int y = 0; y < d.GetLength(1); y++)
                    for (int x = 0; x < d.GetLength(0); x++)
                        foo[x + (d.GetLength(1) * y) + (z * d.GetLength(2))] = d[x, y, z];

            return foo;
        }             


        /// <summary>
        /// Finds the centroid feature of a feature cluster
        /// </summary>
        /// <param name="c">The cluster of feature vectors</param>
        /// <returns></returns>
        static FeatureVector centroid(List<FeatureVector> c, FeatureVector initCentroid)
        {
            FeatureVector fv = new FeatureVector(c[0].vec.Length, -1);
            double[] column = new double[c.Count+1];
            //double[] elements = new double[1];
            /*
            for (int z = 0; z < c[0].markov.GetLength(2); z++)
                for (int y = 0; y < c[0].markov.GetLength(1); y++)
                    for (int x = 0; x < c[0].markov.GetLength(0); x++)
                    {
                            elements = new double[c.Count];
                            for (int i = 0; i < c.Count(); i++) elements[i] = c[i].markov[x, y, z];
                            mv.markov[x, y, z] = m(elements);
                    }
             * */


            for (int x = 0; x < c[0].vec.Length; x++)
            {
                for (int i = 0; i < column.Length-1; i++)
                    column[i] = c[i].vec[x];
                column[column.Length-1] = initCentroid.vec[x];
                fv.vec[x] = m(column);
            }

            return fv;
        }

        /// <summary>
        /// Finds the Absolute Central Moment of a group
        /// </summary>
        /// <param name="p"></param>
        /// <param name="height">The height of the video frame</param>
        /// <param name="width">The width of the video frame</param>
        /// <param name="f"></param>
        /// <param name="mean"></param>
        /// <returns></returns>
        static double AbsCentralMoment(int p, int height, int width, double[] f, double mean)
        {
            double sum = 0;
            for (int i = 0; i < f.Length; i++)
                sum += Math.Pow((Math.Abs(f[i] - mean)), p);
            sum /= (height * width);
            return sum;
        }




        /// <summary>
        /// Reduces the feature dimension of M_Group to (markov_t X markov_t, 2) by reducing the range of coefficients to [0, markov_t]. If any values in the frequency block are above the threshold, then set them to the threshold value.
        /// If Mode 1, also divides all values by 32
        /// </summary>
        /// <param name="T"></param>
        /// <param name="f"></param>
        /// <returns></returns>
        static double[, ,] Truncate(double[, ,] f, int mode)
        {
            int T = markov_t;
            for (int i = 0; i < f.GetLength(2); i++)
                for (int j = 0; j < f.GetLength(1); j++)
                    for (int k = 0; k < f.GetLength(0); k++)
                        if (f[k, j, i] > T || f[k, j, i] < -T)
                        {
                            //if (mode == 1)
                               //f[k, j, i] /= 32;
                            //if (mode == 0)
                                f[k, j, i] = (T * (f[k, j, i] / Math.Abs(f[k, j, i])));
                        }
            return f;
        }

        /// <summary>
        /// Calculates the cosine distance between two feature vectors.
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        static double cosine(FeatureVector a, FeatureVector b)
        {

            double numerator = 0;
            double denominator_a = 0;
            double denominator_b = 0;

            for (int i = 0; i < a.vec.Length; i++)
                numerator += (a.vec[i] * b.vec[i]);

            for (int i = 0; i < a.vec.Length; i++)
                denominator_a += Math.Pow(a.vec[i], 2);

            for (int i = 0; i < a.vec.Length; i++)
                denominator_b += Math.Pow(b.vec[i], 2);

            if (denominator_a == 0 || denominator_b == 0)
                return 1.0;

                return Math.Abs(1 - (numerator / ((Math.Sqrt(denominator_a) * Math.Sqrt(denominator_b)))));
            
            
        }

        /// <summary>
        /// Calculates the Euclidean distance between two feature vectors.
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        static double distance(FeatureVector a, FeatureVector b)
        {
            double numerator = 0;

            for (int i = 0; i < a.vec.Length; i++)
                numerator += Math.Pow(Math.Abs(a.vec[i] - b.vec[i]), 2);

            if (numerator != 0)
                numerator = Math.Pow(numerator, 0.5);

            return numerator;
        }





        /// <summary>
        /// Generates the 888 dimensional feature vector [Markov, abs(p), skew, kurt], p = {1,2,3} essential for classification
        /// *Applies weights (10x) to Kurtosis and Skewness
        /// </summary>
        /// <param name="mv">Markov Object</param>
        /// <param name="abs1">Absolute Central Moment, p = 1</param>
        /// <param name="abs2">Absolute Central Moment, p = 2</param>
        /// <param name="abs3">Absolute Central Moment, p = 3</param>
        /// <param name="skew">Skewness</param>
        /// <param name="kurt">Kurtosis</param>
        /// <returns></returns>
        static FeatureVector featureVector1(Markov mv, double abs1, double abs2, double abs3, double skew, double kurt, int id, double B)
        {
            double[] vec = new double[2 * (((2 * markov_t + 1) * (2 * markov_t + 1))) + 5];
            for (int y = 0; y < (markov_t * 2) + 1; y++) //intra
                for (int x = 0; x < (markov_t * 2) + 1; x++)
                    vec[(((markov_t * 2) + 1) * y) + x] = mv.markov[x, y, 0];
            for (int y = 0; y < (markov_t * 2) + 1; y++) //inter
                for (int x = 0; x < (markov_t * 2) + 1; x++)
                    vec[(((2 * markov_t + 1) * (2 * markov_t + 1))) + (((markov_t * 2) + 1) * y) + x] = mv.markov[x, y, 1];

            vec[((((2 * markov_t + 1) * (2 * markov_t + 1))) + (((2 * markov_t + 1) * (2 * markov_t + 1)))) + 0] = abs1;
            vec[((((2 * markov_t + 1) * (2 * markov_t + 1))) + (((2 * markov_t + 1) * (2 * markov_t + 1)))) + 1] = abs2;
            vec[((((2 * markov_t + 1) * (2 * markov_t + 1))) + (((2 * markov_t + 1) * (2 * markov_t + 1)))) + 2] = abs3;
            vec[((((2 * markov_t + 1) * (2 * markov_t + 1))) + (((2 * markov_t + 1) * (2 * markov_t + 1)))) + 3] = skew;
            vec[((((2 * markov_t + 1) * (2 * markov_t + 1))) + (((2 * markov_t + 1) * (2 * markov_t + 1)))) + 4] = kurt;
            //vec[((((2 * markov_t + 1) * (2 * markov_t + 1))) + (((2 * markov_t + 1) * (2 * markov_t + 1)))) + 5] = B;

            FeatureVector f = new FeatureVector((markov_t * markov_t * 4) + 5, id);

            f.vec = vec; // ur pushing the envelope on ambiguity here chief
            return f;
        }







        /*  
         *  Only thing I didn't write myself but the BEST THING EVER SINCE UNSLICED BREAD:P
         * 
         *  NOT MY CODE!!!!! NOT MY CODE!!!!!
         *  NOT MY CODE!!!!! NOT MY CODE!!!!!
         *  NOT MY CODE!!!!! NOT MY CODE!!!!!
         * 
         *  http://stackoverflow.com/questions/129389/how-do-you-do-a-deep-copy-an-object-in-net-c-specifically
         *  
         *  NOT MY CODE!!!!! NOT MY CODE!!!!!
         *  NOT MY CODE!!!!! NOT MY CODE!!!!!
         *  NOT MY CODE!!!!! NOT MY CODE!!!!!
         * */
        public static T DeepClone<T>(T obj)
        {
            using (var ms = new MemoryStream())
            {
                var formatter = new BinaryFormatter();
                formatter.Serialize(ms, obj);
                ms.Position = 0;

                return (T)formatter.Deserialize(ms);
            }
        }






        /// <summary>
        /// Random initialization k-means with cosine distance
        /// </summary>
        static List<FeatureVector>[] k_means_cosine(FeatureVector[] d, int k, int terminateIn)
        {
            //k means
            FeatureVector[] o = new FeatureVector[d.Length];


            o = DeepClone(d); 
            


            FeatureVector[] means = new FeatureVector[k];
            FeatureVector mean = new FeatureVector(markov_t, markov_t);
            List<FeatureVector>[] clusters = new List<FeatureVector>[k];
            double min = 0;

            bool notDone = true;

            //use random initialization
            o = randomShuffle(o);

            Console.WriteLine("\n ..Clustering with " + k + " randomly selected cluster centers");
            Console.WriteLine("First cluster center selected is: Feature Vector " + means[0].id);
            Console.WriteLine("Second cluster center selected is: Feature Vector " + means[1].id);

            //select k random observations and add them to the cluster centers
            for (int i = 0; i < k; i++)
            {
                //clusters[i].Add(o[i]);
                means[i] = o[i];
            }

            //spherical k means Algorithm
            for (int q = 0; q < terminateIn; q++)
            {
                Console.WriteLine("Iteration: " + q);
                notDone = false;

                //put each element in o[] to the centroid that its closer to
                for (int i = 0; i < o.Length; i++)
                {
                    min = cosine(o[i], means[0]);

                    for (int kk = 0; kk < k; kk++)
                        min = Math.Min(cosine(o[i], means[kk]), min);

                    for (int kk = 0; kk < k; kk++)
                        if (cosine(o[i], means[kk]) == min)
                        {
                            //check if o[i] is not already in the proper cluster
                            if (!clusters[kk].Contains(o[i]))
                            {
                                //o[i] has found a new cluster, move on
                                Console.Write("=");//"Feature Vector " + i + " has moved to cluster " + kk);
                                clusters[kk].Add(o[i]);

                                //remove o[i] from other clusters
                                for (int kkk = 0; kkk < kk; kkk++)
                                    if (kkk != kk)
                                        clusters[kkk].Remove(o[i]);

                                //calculate the new centroid after inserting the new member

                                //make sure everything checks out
                                for (int ui = 0; ui < mean.vec.Length; ui++)
                                    if (!(mean.vec[ui] == means[kk].vec[ui])) notDone = true;
                                means[kk] = mean;

                                /*mean = centroid(clusters[kk]);
                                if (mean != means[kk]) notDone = true;
                                means[kk] = mean;
                                */




                                continue;
                            }
                        }
                }

                
                if (!notDone)
                {
                    Console.WriteLine("\n K means algorithm has been completed after " + q + " steps.");
                    for (int s = 0; s < k; s++)
                    {
                        //Output..
                        Console.WriteLine("= frames in cluster " + s + " are:");
                        for (int ss = 0; ss < clusters[s].Count(); ss++)
                            if (clusters[s][ss].id != -1)
                                Console.WriteLine(ss + ". Feature Vector ( or frame )" + clusters[s][ss].id);
                            
                        //Console.WriteLine("= Centroid for cluster " + s + " is: \n <");

                        //for (int ss = 0; ss < means[k].vec.Length; ss++)
                        //    Console.Write(means[k].vec[ss] + ", ");
                        Console.Write("> \n\n\n");
                    }

                    return clusters;
                }
            }

            Console.WriteLine("Did not converge in specified steps: Terminating.");

            return clusters;
        }



        /// <summary>
        /// k means with cosine distance
        /// 
        /// Use cosine distance (1- cosine similarity)
        /// </summary>
        static List<FeatureVector>[] k_means_cosine(FeatureVector[] d, List<FeatureVector>[] lf, int k, int terminateIn)
        {
            //k means
            FeatureVector[] o = new FeatureVector[d.Length];


            o = DeepClone(d);





            FeatureVector mean = new FeatureVector((2*((2 * markov_t) + 1) * (((2*markov_t)) + 1)) + 5, -1);
            
            List<FeatureVector>[] clusters = new List<FeatureVector>[lf.Length];

            clusters = DeepClone(lf);

            double min = 0;
            int y = 0;
            bool notDone = true;
            FeatureVector[] means = new FeatureVector[k];

            for (int i = 0; i < k; i++)
                means[i] = centroid(clusters[i], lf[i][0]);

            Console.WriteLine("\n SKM Clustering with " + k + " specified cluster centers");


            //k means Algorithm
            for (int q = 0; q < terminateIn; q++)
            {
                Console.Write("\n . \n");
                notDone = false;

                //put each element in o[] to the centroid that its closer to
                for (int i = 0; i < o.Length; i++)
                {
                    min = cosine(o[i], means[0]);
                    y = 0;

                    for (int kk = 0; kk < k; kk++)
                        if (Math.Round(min, 5) != Math.Round(Math.Min(cosine(o[i], means[kk]), min), 5)) y = kk;

                    for (int kk = y; true; )
                    {
                        //check if o[i] is not already in the proper cluster
                        if (!clusters[kk].Contains(o[i]))
                        {
                            //o[i] has found a new cluster, move on
                            Console.WriteLine("Feature Vector " + i + " has moved to cluster " + kk);
                            clusters[kk].Add(o[i]);

                            //remove o[i] from other clusters
                            for (int kkk = 0; kkk < kk; kkk++)
                                if (kkk != kk)
                                    clusters[kkk].Remove(o[i]);

                            //calculate the new centroid after inserting the new member
                            mean = centroid(clusters[kk], lf[kk][0]);
                            //make sure everything checks out
                            for (int ui = 0; ui < mean.vec.Length; ui++)
                                if (!(mean.vec[ui] == means[kk].vec[ui])) notDone = true;
                            means[kk] = mean;

                            //notDone = true;
                            break;
                        }
                        break;
                    }
                    
                }
                if (!notDone)
                {
                    Console.WriteLine("SKM Clustering has been completed after " + q + " steps.");
                    for (int s = 0; s < k; s++)
                    {
                        //Output..
                        Console.WriteLine("Frames in cluster " + s + " are:");
                        for (int ss = 1; ss < clusters[s].Count(); ss++)
                            if (clusters[s][ss].id != -1)
                                Console.Write(clusters[s][ss].id + ", ");

                        Console.WriteLine("\n\n" + (100.0 * (float)(clusters[s].Count() - 1) / (float)((d.Length))) + "% (percent) of the video is in cluster " + s);

                        //Console.WriteLine("= Centroid for cluster " + s + " is: \n <");

                        //for (int ss = 0; ss < means[s].vec.Length; ss++)
                        //    Console.Write(means[s].vec[ss] + ", ");
                        Console.Write("> \n\n");
                    }
                    return clusters;
                }
                

                
            }

            Console.WriteLine("Did not converge in specified steps: Terminating.");

            return clusters;

        }


        /// <summary>
        /// k means with cosine distance
        /// 
        /// Use cosine distance (1- cosine similarity)
        /// </summary>
        static List<FeatureVector>[] k_means(FeatureVector[] d, List<FeatureVector>[] lf, int k, int terminateIn)
        {
            //k means
            FeatureVector[] o = new FeatureVector[d.Length];


            o = DeepClone(d);





            FeatureVector mean = new FeatureVector((2 * ((2 * markov_t) + 1) * (((2 * markov_t)) + 1)) + 5, -1);

            List<FeatureVector>[] clusters = new List<FeatureVector>[lf.Length];

            clusters = DeepClone(lf);

            double min = 0;
            int y = 0;
            bool notDone = true;
            FeatureVector[] means = new FeatureVector[k];

            for (int i = 0; i < k; i++)
                means[i] = centroid(clusters[i], lf[i][0]);

            Console.WriteLine("\n SKM Clustering with " + k + " specified cluster centers");


            //k means Algorithm
            for (int q = 0; q < terminateIn; q++)
            {
                Console.Write("\n . \n");
                notDone = false;

                //put each element in o[] to the centroid that its closer to
                for (int i = 0; i < o.Length; i++)
                {
                    min = distance(o[i], means[0]);
                    y = 0;

                    for (int kk = 0; kk < k; kk++)
                        if (Math.Round(min, 5) != Math.Round(Math.Min(distance(o[i], means[kk]), min), 5)) y = kk;

                    for (int kk = y; true; )
                    {
                        //check if o[i] is not already in the proper cluster
                        if (!clusters[kk].Contains(o[i]))
                        {
                            //o[i] has found a new cluster, move on
                            Console.WriteLine("Feature Vector " + i + " has moved to cluster " + kk);
                            clusters[kk].Add(o[i]);

                            //remove o[i] from other clusters
                            for (int kkk = 0; kkk < kk; kkk++)
                                if (kkk != kk)
                                    clusters[kkk].Remove(o[i]);

                            //calculate the new centroid after inserting the new member
                            mean = centroid(clusters[kk], lf[kk][0]);
                            //make sure everything checks out
                            for (int ui = 0; ui < mean.vec.Length; ui++)
                                if (!(mean.vec[ui] == means[kk].vec[ui])) notDone = true;
                            means[kk] = mean;

                            //notDone = true;
                            break;
                        }
                        break;
                    }

                }
                if (!notDone)
                {
                    Console.WriteLine("SKM Clustering has been completed after " + q + " steps.");
                    for (int s = 0; s < k; s++)
                    {
                        //Output..
                        Console.WriteLine("Frames in cluster " + s + " are:");
                        for (int ss = 1; ss < clusters[s].Count(); ss++)
                            if (clusters[s][ss].id != -1)
                                Console.Write(clusters[s][ss].id + ", ");

                        Console.WriteLine("\n\n" + (100.0 * (float)(clusters[s].Count() - 1) / (float)((d.Length))) + "% (percent) of the video is in cluster " + s);

                        //Console.WriteLine("= Centroid for cluster " + s + " is: \n <");

                        //for (int ss = 0; ss < means[s].vec.Length; ss++)
                        //    Console.Write(means[s].vec[ss] + ", ");
                        Console.Write("> \n\n");
                    }
                    return clusters;
                }



            }

            Console.WriteLine("Did not converge in specified steps: Terminating.");

            return clusters;

        }



        /// <summary>
        /// fuzzy c means with n dimensional vector
        /// 
        /// Use Euclidean distance
        /// </summary>
        static double[,] fuzzy_c_means(FeatureVector[] f, int c, double sigma, FeatureVector[] initialMeans)
        {
            FeatureVector[] o = new FeatureVector[f.Length];

            o = DeepClone(f);

            double m = 1.045;

            FeatureVector[] centroids = new FeatureVector[c];
            //Initialize 
            for (int i = 0; i < c; i++)
                centroids[i] = DeepClone(initialMeans[i]);

            FeatureVector numerator = new FeatureVector(o[0].vec.Length, -1);
            double denominator = 0;


            FeatureVector mean = new FeatureVector(markov_t, markov_t);
            List<FeatureVector>[] clusters = new List<FeatureVector>[c];

            for (int i = 0; i < c; i++)
                clusters[i] = new List<FeatureVector>();



            Console.WriteLine("\n ..Clustering with " + c + " specified cluster centers");


            int n = o.Length;
            int q = 0;
            bool again = false;

            double dem = 0;

            double[,] U_mem = new double[c, n];
            double[,] U_old = new double[c, n];
            double nm = 0;
            double dm = 0;


            //Fuzzy c-means algorithm
            while (true)
            {
                again = false;
                //Store Old U
                for (int i = 0; i < U_mem.GetLength(0); i++)
                    for (int j = 0; j < U_mem.GetLength(1); j++)
                        U_old[i, j] = U_mem[i, j];

                //U
                for (int i = 0; i < U_mem.GetLength(0); i++)
                    for (int j = 0; j < U_mem.GetLength(1); j++)
                    {
                        for (int l = 0; l < c; l++)
                        {

                            nm = distance(DeepClone(o[j]), DeepClone(centroids[i]));
                            dm = distance(DeepClone(o[j]), DeepClone(centroids[l]));
                            if (dm != 0)
                                dem += Math.Pow((nm / dm), 2.0 / (m - 1));
                            //else if (nm == 0)
                            //    dem = 1.0;

                        }

                        if (dem != 0)
                            dem = 1 / dem;


                        U_mem[i, j] = dem;

                        for (int g = 0; g < j; g++)
                        {
                            if (o[g] != centroids[i])
                                continue;
                            if (g == j - 1) // if all vector components match, 
                                //ergo distance == 0 and division by zero would give us NaN then U_mem = 0
                                U_mem[i, j] = 1.0;
                        }

                        dem = 0;
                    }

                // Codebook (centroids[])
                for (int i = 0; i < c; i++)
                {
                    for (int j = 0; j < n; j++)
                        denominator += Math.Pow(U_mem[i, j], m);


                    for (int j = 0; j < n; j++)
                        numerator.vec = DeepClone(assignadd(DeepClone(numerator.vec), DeepClone(mult(DeepClone(o[j].vec), Math.Pow(U_mem[i, j], m)))));

                    if (denominator != 0)
                        centroids[i].vec = DeepClone(mult(DeepClone(numerator.vec), 1 / denominator));
                    else
                        centroids[i].vec = DeepClone(mult(DeepClone(numerator.vec), denominator));


                    denominator = 0;
                    numerator = new FeatureVector(o[0].vec.Length, -1);
                }

                //Compare old U to new U
                for (int i = 0; i < U_mem.GetLength(0); i++)
                    for (int j = 0; j < U_mem.GetLength(1); j++)
                        if (U_old[i, j] - U_mem[i, j] <= -sigma) again = true;
               
                if (!again) break;

                Console.Write(".");
                q++;


                //again..
            }
            Console.WriteLine("\n FCM Converged after " + q + " iterations.");

            //Assemble the cluster collection
            for (int i = 0; i < U_mem.GetLength(0); i++)
                for (int j = 0; j < U_mem.GetLength(1); j++)
                    if (U_mem[i, j] >= 0.5)
                        clusters[i].Add(o[j]);

            //Output U_mem
            for (int i = 0; i < U_mem.GetLength(0); i++)
            {
                Console.WriteLine("Centroid" + i);
                for (int j = 0; j < U_mem.GetLength(1); j++)
                    Console.WriteLine(U_mem[i, j]);
            }

            for (int s = 0; s < c; s++)
            {
                //Output..
                Console.WriteLine("\n\n Feature Vectors in cluster " + s + " are:");
                for (int ss = 0; ss < clusters[s].Count(); ss++)
                    if (clusters[s][ss].id != -1)
                        Console.Write(clusters[s][ss].id + ", ");

                //Console.WriteLine("\n\n" + (100.0 * (float)(clusters[s].Count() - 1) / (float)((f.Length))) + "% (percent) of the video is in cluster " + s);
                //Console.WriteLine("= Centroid for cluster " + s + " is: \n <");

                //for (int ss = 0; ss < initialCentroids[s].vec.Length; ss++)
                //    Console.Write(initialCentroids[s].vec[ss] + ", ");
                Console.Write(" > \n");
            }

            return U_mem;
        }

        /// <summary>
        /// fuzzy c means with n dimensional vector
        /// 
        /// Use Euclidean distance
        /// </summary>
        static double[,] spherical_fuzzy_c_means(FeatureVector[] f, int c, double sigma, FeatureVector[] initialMeans)
        {
            FeatureVector[] o = new FeatureVector[f.Length];

            o = DeepClone(f);

            double m = 1.045;

            FeatureVector[] centroids = new FeatureVector[c];
            //Initialize 
            for (int i = 0; i < c; i++)
                centroids[i] = DeepClone(initialMeans[i]);

            FeatureVector numerator = new FeatureVector(o[0].vec.Length, -1);
            double denominator = 0;


            FeatureVector mean = new FeatureVector(markov_t, markov_t);
            List<FeatureVector>[] clusters = new List<FeatureVector>[c];

            for (int i = 0; i < c; i++)
                clusters[i] = new List<FeatureVector>();



            Console.WriteLine("\n ..Clustering with " + c + " specified cluster centers");


            int n = o.Length;
            int q = 0;
            bool again = false;

            double dem = 0;

            double[,] U_mem = new double[c, n];
            double[,] U_old = new double[c, n];
            double nm = 0;
            double dm = 0;


            //Fuzzy c-means algorithm
            while (true)
            {
                again = false;
                //Store Old U
                for (int i = 0; i < U_mem.GetLength(0); i++)
                    for (int j = 0; j < U_mem.GetLength(1); j++)
                        U_old[i, j] = U_mem[i, j];

                //U
                for (int i = 0; i < U_mem.GetLength(0); i++)
                    for (int j = 0; j < U_mem.GetLength(1); j++)
                    {
                        for (int l = 0; l < c; l++)
                        {

                            nm = cosine(DeepClone(o[j]), DeepClone(centroids[i]));
                            dm = cosine(DeepClone(o[j]), DeepClone(centroids[l]));
                            if (dm != 0)
                                dem += Math.Pow((nm / dm), 2.0 / (m - 1));
                            //else if (nm == 0)
                            //    dem = 1.0;

                        }

                        if (dem != 0)
                            dem = 1 / dem;


                        U_mem[i, j] = dem;
                        
                        /*for (int g = 0; g < j; g++)
                        {
                            if (o[g] != centroids[i])
                                continue;
                            if (g == j - 1) // if all vector components match, 
                                //ergo distance == 0 and division by zero would give us NaN then U_mem = 0
                                //U_mem[i, j] = 1.0;
                        }*/
                        
                        dem = 0;
                    }

                // Codebook (centroids[])
                for (int i = 0; i < c; i++)
                {
                    for (int j = 0; j < n; j++)
                        denominator += Math.Pow(U_mem[i, j], m);


                    for (int j = 0; j < n; j++)
                        numerator.vec = DeepClone(assignadd(DeepClone(numerator.vec), DeepClone(mult(DeepClone(o[j].vec), Math.Pow(U_mem[i, j], m)))));

                    if (denominator != 0)
                        centroids[i].vec = DeepClone(mult(DeepClone(numerator.vec), 1 / denominator));
                    else
                        centroids[i].vec = DeepClone(mult(DeepClone(numerator.vec), denominator));


                    denominator = 0;
                    numerator = new FeatureVector(o[0].vec.Length, -1);
                }

                //Compare old U to new U
                for (int i = 0; i < U_mem.GetLength(0); i++)
                    for (int j = 0; j < U_mem.GetLength(1); j++)
                        if (U_old[i, j] - U_mem[i, j] <= -sigma) again = true;

                if (!again) break;

                Console.Write(".");
                q++;


                //again..
            }
            Console.WriteLine("\n Spherical FCM Converged after " + q + " iterations.");

            //Assemble the cluster collection
            for (int i = 0; i < U_mem.GetLength(0); i++)
                for (int j = 0; j < U_mem.GetLength(1); j++)
                    if (U_mem[i, j] >= 0.5)
                        clusters[i].Add(o[j]);

            //Output U_mem
            for (int i = 0; i < U_mem.GetLength(0); i++)
            {
                Console.WriteLine("Centroid" + i);
                for (int j = 0; j < U_mem.GetLength(1); j++)
                    Console.WriteLine(U_mem[i, j]);
            }

            for (int s = 0; s < c; s++)
            {
                //Output..
                Console.WriteLine("\n\n Feature Vectors in cluster " + s + " are:");
                for (int ss = 0; ss < clusters[s].Count(); ss++)
                    if (clusters[s][ss].id != -1)
                        Console.Write(clusters[s][ss].id + ", ");

                //Console.WriteLine("\n\n" + (100.0 * (float)(clusters[s].Count() - 1) / (float)((f.Length))) + "% (percent) of the video is in cluster " + s);
                //Console.WriteLine("= Centroid for cluster " + s + " is: \n <");

                //for (int ss = 0; ss < initialCentroids[s].vec.Length; ss++)
                //    Console.Write(initialCentroids[s].vec[ss] + ", ");
                Console.Write(" > \n");
            }

            return U_mem;
        }


        static double[] mult(double[] a, double i)
        {
            double[] b = new double[a.Length];
            b = DeepClone(a);
            for (int ii = 0; ii < a.Length; ii++) b[ii] *= i; return b;
        }

        static double[] assignadd(double[] a, double[] b)
        {
            double[] c = new double[a.Length];
            c = DeepClone(a);
            for (int ii = 0; ii < a.Length; ii++) c[ii] += b[ii]; return c;
        }

        /*
         * I h8 C# because it won't lemme overload assignment operators like +=
         * 
        static double[] operator + (double[] a, double[] b)
        {
            double[] c = new double[Math.Max(a.Length, b.Length)];
            for(int i = 0; i < c.Length; i++)
                c[i] = a[i] + b[i];
            return c;
        }
        */

        /// <summary>
        /// Pseudo-randomly shuffles a feature cluster with divide and conquer
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        static FeatureVector[] randomShuffle(FeatureVector[] a)
        {
            int[] u = new int[a.Length];
            int n = 0;
            FeatureVector o = new FeatureVector((markov_t * markov_t) + 5, -1);
            for (int i = 0; i < u.Length; i++)
                u[i] = i; //populate
            randomSelect(u, ref n, ref u);

            for (int i = 0; i < u.Length; i++)
            {
                o = a[i];
                a[i] = a[u[i]]; // rawr
                a[u[i]] = o;
            }

            return a;
        }


        private static void randomSelect(int[] a, ref int n, ref int[] u)
        {

            if (a.Length == 1)
            {
                u[n] = a[0];
                n++;
                return;
            }

            Random r = new Random();
            int pivot = r.Next(1, a.Length - 1);
            u[n] = a[pivot];
            n++;

            int[] c = new int[pivot];
            int[] d = new int[a.Length - pivot - 1];
            Array.Copy(a, c, pivot);
            Array.Copy(a, pivot + 1, d, 0, a.Length - pivot - 1);

            if (c.Length > 0)
                randomSelect(c, ref n, ref u);
            if (d.Length > 0)
                randomSelect(d, ref n, ref u);

            return;
        }


        /// <summary>
        /// 
        /// Returns a state transition tensor (multidimensional array) with
        /// the markov features in the horizontal, vertical, and main diagonal directions
        /// as well as markov features in the time direction
        /// 
        /// 
        /// </summary>
        /// <param name="f"> The output of MGroup(), which is a Rectangular Prism Y(width, height, depth) : Y(FrameWidth, FrameHeight, 4) 
        /// representing 3d DCT coefficients for the current group</param>
        /// <returns>A transition tensor (double[,,) where the x and y positions are average intra-frame markov features, and z position is inter-frame markov feature</returns>
        static double[, ,] transition_Tensor(double[, ,] f, int mode)
        {
            int t1 = markov_t;
            int t2 = markov_t;
            //int a = 0; int b = 0;
            /*if (mode == 0)
            {
                a = 2; b = 3;
            }

            if (mode == 1)
            {
                a = 0; b = 1;
            }*/

            //For a 4x4x4 cube of 3D DCT frequencies f[x,y,z]:

            //
            //
            //
            //      EXPLANATION
            //
            //
            //
            //      Position [a,b,c] of the tensor represents:
            //      a is the row position of the transition tensor
            //      b is the column position of the transition tensor
            //      c is the z position in the transition tensor (inter frame)
            //
            //      Where kron(*) casts the result of a binary statement * to a bit {0,1}
            //
            ///     
            ///     Implementation:
            /// 
            /// [1] STEGANALYSIS OF YOUTUBE COMPRESSED VIDEO USING HIGH-ORDER STATISTICS
            /// IN 3D DCT DOMAIN
            /// Hong Zhao1, 2, Hongxia Wang1, Hafiz
            ///
            /// Info:
            /// 
            /// [2] JPEG Image Steganalysis Utilizing both Intrablock
            /// and Interblock Correlations
            /// 

            double[, ,] markov = new double[(t1*2)+1, (t2*2)+1, 2];
            double horizontal = 0;
            double vertical = 0;
            double diagonal = 0;
            double inter = 0;
            double count = 0;


            for (int m = -t2; m <= t2; m++) //-T to T
            {
                for (int n = -t1; n <= t1; n++) //-T to T
                {
                    for (int i = 0; i < f.GetLength(1); i++) 
                    {
                        for (int j = 0; j < f.GetLength(0); j++) 
                        {

                            if (j < f.GetLength(0) - 1)
                                horizontal += (((int)Math.Round(f[j + 1, i, 1]) == n && (int)Math.Round(f[j, i, 1]) == m) ? 1.0 : 0.0);

                            if (i < f.GetLength(1) - 1)
                                vertical += (((int)Math.Round(f[j, i + 1, 1]) == n && (int)Math.Round(f[j, i, 1]) == m) ? 1.0 : 0.0);

                            if (j < f.GetLength(0) - 1 && i < f.GetLength(1) - 1)
                                diagonal += (((int)Math.Round(f[j + 1, i + 1, 1]) == n && (int)Math.Round(f[j, i, 1]) == m) ? 1.0 : 0.0);

                            if (j < f.GetLength(0) - 1 && i < f.GetLength(1) - 1)
                                inter += (((int)Math.Round(f[j, i, 0]) == n && (int)Math.Round(f[j, i, 1]) == m) ? 1.0 : 0.0);

                            if ((int)Math.Round(f[j, i, 1]) == m) count++;


                        }
                    }
                    //inter frame markov feature :D
                    if (count != 0)
                        markov[m + t1, n + t2, 1] = (inter / count);
                    else
                        markov[m + t1, n + t2, 1] = 0; //m is not a possible value in the group

                    // intra frame markov feature
                    if (count != 0)
                        markov[m + t1, n + t2, 0] = ((horizontal + vertical + diagonal) / count / 3.0);
                    else
                        markov[m + t1, n + t2, 0] = 0; //m is not a possible value in the group


                    horizontal = 0;
                    vertical = 0;
                    diagonal = 0;
                    inter = 0;
                    count = 0;
                }
            }

            return markov;
        }

        /*
        /// <summary>
        /// Helper 
        /// </summary>
        /// <param name="f"></param>
        /// <param name="num"></param>
        /// <returns></returns>
        static double[] _f(Frequency_4x4x4 f, int i)
        {
            double[] d = { f.frequencies[0, 0, i], f.frequencies[0, 1, i], f.frequencies[0, 2, i], f.frequencies[0, 3, i], f.frequencies[1, 0, i], f.frequencies[1, 1, i], f.frequencies[1, 2, i], f.frequencies[1, 3, i], f.frequencies[2, 0, i], f.frequencies[2, 1, i], f.frequencies[2, 2, i], f.frequencies[2, 3, i], f.frequencies[3, 0, i], f.frequencies[3, 1, i], f.frequencies[3, 2, i], f.frequencies[3, 3, i] };
            return d; 
        }



        /// <summary>
        /// Helper 
        /// </summary>
        /// <param name="f"></param>
        /// <param name="num"></param>
        /// <returns></returns>
        static Frequency_4x4x4 f_(Frequency_4x4x4 f,  double[] f0, int i)
        {
            Frequency_4x4x4 f1 = new Frequency_4x4x4();
            Array.Copy( f.frequencies, f1.frequencies, f1.frequencies.Length);

            f1.frequencies[0, 0, i] = f.frequencies[0, 0, i];
            f1.frequencies[0, 1, i] = f.frequencies[0, 1, i];
            f1.frequencies[0, 2, i] = f.frequencies[0, 2, i];
            f1.frequencies[0, 3, i] = f.frequencies[0, 3, i];
            f1.frequencies[1, 0, i] = f.frequencies[1, 0, i];
            f1.frequencies[1, 1, i] = f.frequencies[1, 1, i];
            f1.frequencies[1, 2, i] = f.frequencies[1, 2, i];
            f1.frequencies[1, 3, i] = f.frequencies[1, 3, i];
            f1.frequencies[2, 0, i] = f.frequencies[2, 0, i];
            f1.frequencies[2, 1, i] = f.frequencies[2, 1, i];
            f1.frequencies[2, 2, i] = f.frequencies[2, 2, i];
            f1.frequencies[2, 3, i] = f.frequencies[2, 3, i];
            f1.frequencies[3, 0, i] = f.frequencies[3, 0, i];
            f1.frequencies[3, 1, i] = f.frequencies[3, 1, i];
            f1.frequencies[3, 2, i] = f.frequencies[3, 2, i];
            f1.frequencies[3, 3, i] = f.frequencies[3, 3, i];

            return f1;
        }
         * 
         * 
        */

        static Bitmap median(Bitmap b)
        {
            Bitmap c = DeepClone(b);
            int[] list = new int[9];

            for (int y = 0; y < b.Height; y++)
                for (int x = 0; x < b.Width; x++)
                {
                    if (!(x > b.Width - 2) && !(y > b.Height - 2) && !(y < 1) && !(x < 1))
                        for (int i = 0; i < 9; i++)
                            list[i] = b.GetPixel(x + (i % 3) - 1, y + (i / 3) - 1).ToArgb();

                    c.SetPixel(x, y, Color.FromArgb(list.OrderBy(a => a, new CompInt()).ToArray<int>()[4]));
                }
            return c;
        }

        static Bitmap averaging(Bitmap b)
        {
            Bitmap c = DeepClone(b);
            double sum = 0;

                for (int y = 0; y < b.Height; y++)
                    for (int x = 0; x < b.Width; x++)
                    {
                        sum = 0;
                        if (!(x > b.Width - 2) && !(y > b.Height - 2) && !(y < 1) && !(x < 1))
                            for (int i = 0; i < 9; i++)
                                sum += b.GetPixel(x + (i % 3) - 1, y + (i / 3) - 1).ToArgb();
                        
                        sum /= 9;
                        c.SetPixel(x, y, Color.FromArgb((int)sum));
                    }
                return c;
        }

        //secure pseudo random number whatever... see Aaron Sharp
        static Bitmap discrete_SPRNG(Bitmap b)
        {
            Bitmap c = DeepClone(b);
            
            Random rand = new Random();
            int PRN = 0;
            for (int y = 0; y < b.Height; y+=4)
                for (int x = 0; x < b.Width; x += 4)
                {
                    for (int i = 0; i < 16; i++)
                        c.SetPixel(x + (i % 4), y + (i / 4), Color.FromArgb((int)Math.Pow(b.GetPixel(x + (i % 4), y + (i / 4)).ToArgb(), 0.99 + (double)(PRN / 100000.0))));
                    PRN = rand.Next(0, 2000);
                }
            return c;
        }

        static Bitmap gaussian(Bitmap b)
        {
            int[,] kernel = 
            {{2,4,5,4,2},
            {4,9,12,9,4},
            {5, 12, 15, 12, 5},
            {4,9,12,9,4},
            {2,4,5,4,2}
            };

            Bitmap c = DeepClone(b);
            double sum = 0;

            for (int y = 0; y < b.Height; y++)
                for (int x = 0; x < b.Width; x++)
                {
                    sum = 0;
                    if (!(x > b.Width - 3) && !(y > b.Height - 3) && !(y < 2) && !(x < 2))
                    {
                        for (int i = 0; i < 25; i++)
                            sum += kernel[(i / 5), (i % 5)] * b.GetPixel(x + (i % 5) - 2, y + (i / 5) - 2).ToArgb();
                    }

                    sum /= 159;
                    c.SetPixel(x, y, Color.FromArgb((int)Math.Round(sum)));
                }
            return c;
        }

        /// <summary>
        /// horizontal sobel operator
        /// </summary>
        /// <param name="b"></param>
        /// <returns></returns>
        static int sobelh(int[,] b, int x, int y)
        {

            int[,] kernel = 
            {
            {-1,0,1},
            {-2,0,2},
            {-1,0,1},
            };

            double sum = 0;

            if (!(x > b.GetLength(1) - 2) && !(y > b.GetLength(0) - 2) && !(y < 1) && !(x < 1))
                for (int i = 0; i < 9; i++)
                    sum += kernel[(i / 3), (i % 3)] * b[x + (i % 3) - 1, y + (i / 3) - 1];

            return (int)Math.Round(sum);
                
        }

        /// <summary>
        /// vertical sobel operator
        /// </summary>
        /// <param name="b"></param>
        /// <returns></returns>
        static int sobelv(int[,] b, int x, int y)
        {
            int[,] kernel = 
            {
            {-1,-2,-1},
            { 0, 0, 0},
            { 1, 2, 1},
            };

            double sum = 0;

            if (!(x > b.GetLength(1) - 2) && !(y > b.GetLength(0) - 2) && !(y < 1) && !(x < 1))
                for (int i = 0; i < 9; i++)
                    sum += kernel[(i / 3), (i % 3)] * b[x + (i % 3) - 1, y + (i / 3) - 1];

            return (int)Math.Round(sum);
                
        }

        /// <summary>
        /// vertical blocking artifact coefficient
        /// </summary>
        /// <param name="b"></param>
        /// <returns></returns>
        static double AC_ver(int[,] b, int m, int n)
        {
            double c = 0;
            if (b_vin(b, m, n) == 0)
                c = b_vb(b, m, n) * 0.25;
            else
                c = b_vb(b, m, n) / b_vin(b, m, n);
            return c;
        }

        /// <summary>
        /// horizontal blocking artifact coefficient
        /// </summary>
        /// <param name="b"></param>
        /// <returns></returns>
        static double AC_hor(int[,] b, int m, int n)
        {
            double c = 0;
            if (b_hin(b, m, n) == 0)
                c = b_hb(b) * 0.25;
            else
                c = b_hb(b) / b_hin(b, m, n);
            return c;
        }

        /// <summary>
        /// average gradient of horizontal block boundary 
        /// </summary>
        /// <param name="b"></param>
        /// <returns></returns>
        static double b_hb(int[,] b)
        {
            double sum = 0;
            for (int i = 1; i < 5; i++)
                sum += sobelh(b, 0, i);
            for (int i = 1; i < 5; i++)
                sum += sobelh(b, 3, i);
            sum  = sum/8.0;
            return sum;
        }

        /// <summary>
        /// horizontal zero gradient
        /// </summary>
        /// <param name="b"></param>
        /// <returns></returns>
        static int Z_hor(int[,] b, int x, int y)
        {
            return sobelh(b, x, y) == 0 ? 1 : 0;
        }

        /// <summary>
        /// vertical zero gradient
        /// </summary>
        /// <param name="b"></param>
        /// <returns></returns>
        static int Z_ver(int[,] b, int x, int y)
        {
            return sobelv(b, x, y) == 0 ? 1 : 0;
        }

        /// <summary>
        /// average gradient of horizontal block interior
        /// </summary>
        /// <param name="b"></param>
        /// <returns></returns>
        static double b_hin(int[,] b, int m, int n)
        {
            double sum = 0;
            for (int i = 2; i < 4; i++)
                for (int j = 2; j < 4; j++)
                sum += sobelh(b, i, j);
            
            sum = sum / 4.0;
            return sum;
        }

        /// <summary>
        /// average gradient of vertical block boundary, etc..
        /// </summary>
        /// <param name="b"></param>
        /// <returns></returns>
        static double b_vb(int[,] b, int m, int n)
        {
            double sum = 0;
            for (int i = 1; i < 5; i++)
                sum += sobelv(b, i, 0);
            for (int i = 1; i < 5; i++)
                sum += sobelv(b, i, 3);
            sum = sum / 8.0;
            return sum;
        }

        /// <summary>
        /// average gradient of vertical block interior
        /// </summary>
        /// <param name="b"></param>
        /// <returns></returns>
        static double b_vin(int[,] b, int m, int n)
        {
            double sum = 0;
            for (int i = 2; i < 4; i++)
                for (int j = 2; j < 4; j++)
                    sum += sobelv(b, i, j);

            sum = sum / 4.0;
            return sum;
        }

        /// <summary>
        /// horizontal flatness coefficient
        /// </summary>
        /// <returns></returns>
        static double fc_hor(int[,] b, int m, int n)
        {
            double sum = 0;
            for (int i = 1; i < 5; i++)
                for (int j = 1; j < 5; j++)
                    sum += Z_hor(b, j, i);
            sum *= (4.5 / 16);
            return sum;
        }

        /// <summary>
        /// ..vertical flatness coefficient
        /// </summary>
        /// <returns></returns>
        static double fc_ver(int[,] b, int m, int n)
        {
            double sum = 0;
            for (int i = 1; i < 5; i++)
                for (int j = 1; j < 5; j++)
                    sum += Z_ver(b, j, i);
            sum *= (4.5 / 16);
            return sum;
        }

        /// <summary>
        /// ..from paper
        /// </summary>
        static double 
            sobel_magnitude(int[,] b, int x, int y)
        {
            return Math.Sqrt(Math.Pow(sobelh(b, x, y), 2) + Math.Pow(sobelv(b, x, y), 2));
        }

        /// <summary>
        /// ..mean block edge strength
        /// </summary>
        /// <param name="?"></param>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <returns></returns>
        static double 
            es(int[,] b, int m, int n)
        {
            double sum = 0;
            for (int i = 1; i < 5; i++)
                for (int j = 1; i < 5; i++)
                    sum += sobel_magnitude(b, j, i);
            sum /= 16;
            return sum;
        }
        /// <summary>
        /// distribution density
        /// </summary>
        /// <param name="?"></param>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <returns></returns>
        static double 
            distdens(int[,] b, int m, int n, int gamma)
        {
            double sum = 0;
            for (int i = 1; i < 5; i++)
                for (int j = 1; i < 5; i++)
                    sum += (sobel_magnitude(b, j ,i) > gamma) ? 0 : 1;
            sum /= 16;
            return sum;
        }

        /// <summary>
        /// ..randomness of texture
        /// </summary>
        /// <param name="?"></param>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <returns></returns>
        static double 
            trc(int[,] b, int m, int n, int gamma)
        {
            return es(b, m, n) * distdens(b, m, n, 35);
        }

        /// <summary>
        /// luminance adaptation..
        /// </summary>
        /// <param name="?"></param>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <returns></returns>
        static double 
            lac(int[,] b, int m, int n)
        {
            double sum = 0;
            for (int i = 1; i < 5; i++)
                for (int j = 1; j < 5; j++)
                    sum += b[j, i];
            sum /= 16;
            return (sum <= 71) ? 1.152 * Math.Log(1 + Math.Sqrt(sum), Math.E) 
                : 1.152 * Math.Log(1 + Math.Sqrt(255 - sum), Math.E);
        }

        /// <summary>
        /// block sensitivity coefficient
        /// </summary>
        /// <param name="?"></param>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <returns></returns>
        static double 
            sc(int[,] b, int m, int n, double trc_m)
        {
            //you'll notice a slight change, I neglect the / 2... speeeeeeeeeeeeeed.
            return (AC_hor(b, m, n) + AC_ver(b, m, n)) < (fc_hor(b, m, n) + fc_ver(b, m, n)) ? lac(b, m, n) : 
                lac(b, m, n) * tmc(b, m, n, 35);
                
        }


        /// <summary>
        /// randomness of texture
        /// </summary>
        /// <param name="?"></param>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <returns></returns>
        static double
            tmc(int[,] b, int m, int n, double trc_max)
        {
            //you'll notice a slight change, I neglect the / 2... speeeeeeeeeeeeeed.
            return 1/Math.Pow((1 + trc(b, m, n, 35)/trc_max), 0.5);

        }

        /// <summary>
        /// Takes a value between 0 and 1 and 
        /// converts it to a location on the visible light spectrum. (Blue to Red)
        /// </summary>
        /// <param name="d"></param>
        /// <returns></returns>
        static byte[] spectrum(double d)
        {
            //[R][G][B] [0][1][2]
            int col = (int)(d * 765);
            int red = 0;
            byte[] RGB = { 0, 0, 0 };
            RGB[0] = ((col - 255)) > 0 ? (byte)((col - 255)) : (byte)0;
            RGB[1] = (col - 127) > 0 ? (byte)(col - 127) : (byte)0;
            RGB[2] = (col      ) > 0 ? (byte)(col      ) : (byte)0;
             
            RGB[2] = (byte)(127 - Math.Abs(RGB[2] - 127));
            RGB[1] = (byte)(127 - Math.Abs(RGB[1] - 127));
            

            return RGB;
        }
        /// <summary>
        /// Returns a value representing the "blockiness" of an intensity image.
        /// </summary>
        /// <param name="b"></param>
        static double
           S_wsbm(Bitmap b)
        {
            int [,] pixels = new int[6,6];
            Bitmap c = DeepClone(b);
            Bitmap e = DeepClone(b);
            Bitmap d = new Bitmap(b.Width + 2, b.Height + 2);
            double ws_block = 0;

            for (int i = 1; i < e.Height - 1; i++)
                for (int j = 1; j < e.Width - 1; j++)
                    d.SetPixel(j, i, e.GetPixel(j - 1, i - 1)); 

            double trc_m = trc_max(e);

            for (int i = 0; i < b.Height; i++)
                d.SetPixel(0, i, c.GetPixel(0, i));

            for (int i = 0; i < b.Height; i++)
                d.SetPixel(b.Width - 1, i, c.GetPixel(b.Width - 1, i));

            for (int i = 0; i < b.Width; i++)
                d.SetPixel(i, b.Height - 1, c.GetPixel(i, b.Height - 1));

            for (int i = 0; i < b.Width; i++)
                d.SetPixel(i, 0, c.GetPixel(i, 0));
            
                    
            for (int i = 0; i + 6 < b.Height; i+=4)
                for (int j = 0; j + 6 < b.Width; j+=4)
                {
                    for (int y = 0; y< 6; y++)
                        for (int x = 0; x < 6; x++)
                            pixels[x, y] = d.GetPixel(x + j, y + i).ToArgb();

                    //neglected the / 2 in comparison
                    if ((AC_ver(pixels, j, i) + AC_hor(pixels, j, i)) < (fc_ver(pixels, j, i) + fc_hor(pixels, j, i)))
                        ws_block += (sc(pixels, i, j, trc_m) * (fc_hor(pixels, i, j) + fc_ver(pixels, i, j))/2);
                    else
                        ws_block += (sc(pixels, i, j, trc_m) * (AC_ver(pixels, j, i) + AC_hor(pixels, j, i))/2);

                }

            return ws_block / (b.Height * b.Width / 16);

             
        }

        /// <summary>
        /// Returns a value representing the "blockiness" of an intensity image.
        /// </summary>
        /// <param name="b"></param>
        static int
           trc_max(Bitmap b)
        {
            int[,] pixels = new int[6, 6];
            Bitmap c = DeepClone(b);
            Bitmap e = DeepClone(b);
            Bitmap d = new Bitmap(b.Width + 2, b.Height + 2);
            
            int max = 0;

            for (int i = 1; i < e.Height - 1; i++)
                for (int j = 1; j < e.Width - 1; j++)
                    d.SetPixel(j, i, e.GetPixel(j - 1, i - 1)); 

            for (int i = 0; i < b.Height; i++)
                d.SetPixel(0, i, c.GetPixel(0, i));

            for (int i = 0; i < b.Height; i++)
                d.SetPixel(b.Width - 1, i, c.GetPixel(b.Width - 1, i));

            for (int i = 0; i < b.Width; i++)
                d.SetPixel(i, b.Height - 1, c.GetPixel(i, b.Height - 1));

            for (int i = 0; i < b.Width; i++)
                d.SetPixel(i, 0, c.GetPixel(i, 0));
            

            for (int i = 1; i + 6 < b.Height; i += 4)
                for (int j = 1; j + 6 < b.Width; j += 4)
                {
                    for (int y = 0; y < 6; y++)
                        for (int x = 0; x < 6; x++)
                            pixels[x , y] = d.GetPixel(x + j, y + i).ToArgb();

                    if (Math.Max(trc(pixels, j, i, 35), max) != trc(pixels, max % b.Width, max / b.Height, 35))
                        max = (i * b.Width) + j;
                    //neglected the / 2 in comparison
                    
                }

            return max;


        }

        static void Main(string[] args)
        {
            byte[] videoData = new byte[0];
            byte[] bytes = new byte[0];
            Bitmap b = new Bitmap(1, 1);
            Bitmap c = new Bitmap(1, 1);
            Bitmap b1 = new Bitmap(1, 1);
            Bitmap b2 = new Bitmap(1, 1);
            Bitmap b3 = new Bitmap(1, 1);

            string savefilename = "file";
            byte[] saveFile = new byte[1];
            Frequency_4x4x4 steg = new Frequency_4x4x4();

            Console.WriteLine("XMAS PACKAGER -- (0x00)");
            Console.WriteLine("------------");
            Console.WriteLine("Digital Data Hiding for Online Videos");
            Console.WriteLine("What is the video width?");
            int videoWidth = Int32.Parse(Console.ReadLine());
            Console.WriteLine("What is the video height?");
            int videoHeight = Int32.Parse(Console.ReadLine());
            Console.WriteLine("What is the number of frames of the video?");
            int frameLength = Int32.Parse(Console.ReadLine());
            Console.WriteLine("What is the video's filename?");
            string filename = Console.ReadLine();
            Console.WriteLine("What is the stegano file's filename?");
            string sfilename = Console.ReadLine();
            Console.WriteLine("Are you hiding data, attacking it, reconstructing it, or performing steganalysis? <H/R/S/A>");
            string hiding = Console.ReadLine();
            double mse = 0;
            if (hiding == "A")
            {
                Console.WriteLine("\n Attacking Algorithm?");
                Console.WriteLine("1. Averaging Filter 2. Gaussian Filter 3. DST 4. Barrage Noise 5. Median Filter 6. Median of DST 7. Averaging of DST 8. Gaussian of DST 9. Wavelet HL-LH-HH Filter 10. Wavelet HH Filter");
                int resp = Int32.Parse(Console.ReadLine());
                byte[] barrage = new byte[videoWidth * videoHeight / 4 / 4];
                Random r = new Random();
                Console.WriteLine("PSNR for each frame:");
                for (int i = 0; i < frameLength; i++)
                {
                    if (resp == 1)
                    {

                        filesizespecified = bytes.Length;

                        
                        videoData = GetBytesFromFile(filename, (((videoWidth * videoHeight) * i) + (((videoWidth * videoHeight) * i) / 2)), videoWidth * videoHeight);
                        b = BitmapFromArray1DSafe(videoData, videoWidth, videoHeight);
                        c = averaging(b);
                        mse = MSE(b, c);
                        Console.WriteLine(PSNR(mse));
                        videoData = Array1DFromBitmap(c, videoWidth * videoHeight);
                        SaveBytesToFile(filename, (((videoWidth * videoHeight) * i) + (((videoWidth * videoHeight) * i) / 2)), videoWidth * videoHeight, videoData);

                    }
                    if (resp == 2)
                    {

                        filesizespecified = bytes.Length;

                        videoData = GetBytesFromFile(filename, (((videoWidth * videoHeight) * i) + (((videoWidth * videoHeight) * i) / 2)), videoWidth * videoHeight);
                        b = BitmapFromArray1DSafe(videoData, videoWidth, videoHeight);
                        c = gaussian(b);
                        mse = MSE(b, c);
                        Console.WriteLine(PSNR(mse));
                        videoData = Array1DFromBitmap(c, videoWidth * videoHeight);
                        SaveBytesToFile(filename, (((videoWidth * videoHeight) * i) + (((videoWidth * videoHeight) * i) / 2)), videoWidth * videoHeight, videoData);


                    }
                    if (resp == 3)
                    {

                        filesizespecified = bytes.Length;

                        videoData = GetBytesFromFile(filename, (((videoWidth * videoHeight) * i) + (((videoWidth * videoHeight) * i) / 2)), videoWidth * videoHeight);
                        b = BitmapFromArray1DSafe(videoData, videoWidth, videoHeight);
                        c = discrete_SPRNG(b);
                        mse = MSE(b, c);
                        Console.WriteLine(PSNR(mse));
                        videoData = Array1DFromBitmap(c, videoWidth * videoHeight);
                        SaveBytesToFile(filename, (((videoWidth * videoHeight) * i) + (((videoWidth * videoHeight) * i) / 2)), videoWidth * videoHeight, videoData);

                    }
                    if (resp == 4)
                    {

                        filesizespecified = bytes.Length;
                        
                        for (int ik = 0; ik < barrage.Length; ik++)
                            barrage[ik] = (byte)r.Next(0, 256);

                            videoData = GetBytesFromFile(filename, (((videoWidth * videoHeight) * i) + (((videoWidth * videoHeight) * i) / 2)), videoWidth * videoHeight);
                            b = BitmapFromArray1DSafe(videoData, videoWidth, videoHeight);
                            c = DeepClone(b); 
                            b = barrage_Noise(b, 10, barrage, 1);
                            mse = MSE(b, c);
                            Console.WriteLine(PSNR(mse));
                            videoData = Array1DFromBitmap(b, videoWidth * videoHeight);
                            SaveBytesToFile(filename, (((videoWidth * videoHeight) * i) + (((videoWidth * videoHeight) * i) / 2)), videoWidth * videoHeight, videoData);
                        
                    }
                    if (resp == 5)
                    {

                        filesizespecified = bytes.Length;

                        videoData = GetBytesFromFile(filename, (((videoWidth * videoHeight) * i) + (((videoWidth * videoHeight) * i) / 2)), videoWidth * videoHeight);
                        b = BitmapFromArray1DSafe(videoData, videoWidth, videoHeight);
                        c = median(b);
                        mse = MSE(b, c);
                        Console.WriteLine(PSNR(mse));
                        videoData = Array1DFromBitmap(c, videoWidth * videoHeight);
                        SaveBytesToFile(filename, (((videoWidth * videoHeight) * i) + (((videoWidth * videoHeight) * i) / 2)), videoWidth * videoHeight, videoData);

                    }
                    if (resp == 6)
                    {

                        filesizespecified = bytes.Length;

                        videoData = GetBytesFromFile(filename, (((videoWidth * videoHeight) * i) + (((videoWidth * videoHeight) * i) / 2)), videoWidth * videoHeight);
                        b = BitmapFromArray1DSafe(videoData, videoWidth, videoHeight);
                        c = discrete_SPRNG(b);
                        c = median(c);
                        mse = MSE(b, c);
                        Console.WriteLine(PSNR(mse));
                        videoData = Array1DFromBitmap(c, videoWidth * videoHeight);
                        SaveBytesToFile(filename, (((videoWidth * videoHeight) * i) + (((videoWidth * videoHeight) * i) / 2)), videoWidth * videoHeight, videoData);

                    }
                    if (resp == 7)
                    {

                        filesizespecified = bytes.Length;

                        videoData = GetBytesFromFile(filename, (((videoWidth * videoHeight) * i) + (((videoWidth * videoHeight) * i) / 2)), videoWidth * videoHeight);
                        b = BitmapFromArray1DSafe(videoData, videoWidth, videoHeight);
                        c = discrete_SPRNG(b);
                        c = averaging(c);
                        mse = MSE(b, c);
                        Console.WriteLine(PSNR(mse));
                        videoData = Array1DFromBitmap(c, videoWidth * videoHeight);
                        SaveBytesToFile(filename, (((videoWidth * videoHeight) * i) + (((videoWidth * videoHeight) * i) / 2)), videoWidth * videoHeight, videoData);

                    }
                    if (resp == 8)
                    {

                        filesizespecified = bytes.Length;

                        videoData = GetBytesFromFile(filename, (((videoWidth * videoHeight) * i) + (((videoWidth * videoHeight) * i) / 2)), videoWidth * videoHeight);
                        b = BitmapFromArray1DSafe(videoData, videoWidth, videoHeight);
                        c = discrete_SPRNG(b);
                        c = gaussian(c);
                        mse = MSE(b, c);
                        Console.WriteLine(PSNR(mse));
                        videoData = Array1DFromBitmap(c, videoWidth * videoHeight);
                        SaveBytesToFile(filename, (((videoWidth * videoHeight) * i) + (((videoWidth * videoHeight) * i) / 2)), videoWidth * videoHeight, videoData);

                    }
                    if (resp == 9)
                    {

                        filesizespecified = bytes.Length;
                        double[,] d = new double[videoWidth, videoHeight];
                        double[,] d_ = new double[videoWidth, videoHeight];

                        videoData = GetBytesFromFile(filename, (((videoWidth * videoHeight) * i) + (((videoWidth * videoHeight) * i) / 2)), videoWidth * videoHeight);
                        b = BitmapFromArray1DSafe(videoData, videoWidth, videoHeight);
                        c = DeepClone(b);

                        d = Wavelet_Daub(b);
                        
                        //Filter HH, HL, and LH
                        for (int ii = 0; ii < videoHeight/ 2; ii++)
                            for (int jj = 0; jj < videoWidth / 2; jj++)
                                d_[jj, ii] = d[jj, ii];
                            
                        
                        c = iWavelet_Daub(d_);

                        mse = MSE(b, c);
                        Console.WriteLine(PSNR(mse));
                        videoData = Array1DFromBitmap(c, videoWidth * videoHeight);
                        SaveBytesToFile(filename, (((videoWidth * videoHeight) * i) + (((videoWidth * videoHeight) * i) / 2)), videoWidth * videoHeight, videoData);

                    }
                    if (resp == 10)
                    {

                        filesizespecified = bytes.Length;
                        double[,] d = new double[videoWidth, videoHeight];
                        //double[,] d_ = new double[videoWidth, videoHeight];

                        videoData = GetBytesFromFile(filename, (((videoWidth * videoHeight) * i) + (((videoWidth * videoHeight) * i) / 2)), videoWidth * videoHeight);
                        b = BitmapFromArray1DSafe(videoData, videoWidth, videoHeight);
                        c = DeepClone(b);

                        d = Wavelet_Daub(b);

                        //Filter HH
                        for (int ii = videoHeight / 2; ii < videoHeight; ii++)
                            for (int jj = videoWidth / 2; jj < videoWidth; jj++)
                                d[jj, ii] = 0;


                        c = iWavelet_Daub(d);

                        mse = MSE(b, c);
                        Console.WriteLine(PSNR(mse));
                        videoData = Array1DFromBitmap(c, videoWidth * videoHeight);
                        SaveBytesToFile(filename, (((videoWidth * videoHeight) * i) + (((videoWidth * videoHeight) * i) / 2)), videoWidth * videoHeight, videoData);

                    }
                }
                Console.ReadLine();
            }


            if (hiding == "S")
            {

                Console.WriteLine("\n > == Blind Steganalysis using the 3D DCT's high frequencies and K-means Clustering ==");
                Console.WriteLine("\n 1. Specify Cluster Centroids (Recommended) \n 2. CPU-Generated Random Cluster Centroids \n ");
                Bitmap markovian = new Bitmap(1+(((2 * markov_t) + 1)*2), (2 * markov_t) + 1);

                string cent = Console.ReadLine();
                
                if (cent == "2")
                {
                    Console.WriteLine("The CPU will generate k random cluster centroids. This may or may not give the best output. \nWhat is k?");
                    int k = Int32.Parse(Console.ReadLine());
                }

                if (cent == "1")
                {
                    Console.WriteLine("To initialize the k means algorithm, you need to select two representative videos so that the program has an understanding of the difference between the two classes. "
                    + "\n\n The first video should have data hidden inside of all frames (Stegano), while the second should have no data hidden inside of any frame. (Clean) \nWhat is k?");
                    int k = Int32.Parse(Console.ReadLine());
                    Console.WriteLine("Specify the filename of the stegano video");
                    string steganovid = Console.ReadLine();
                    Console.WriteLine("Specify the filename of the clean video");
                    string cleanvid = Console.ReadLine();
                    Console.WriteLine("What is GOP? (Set this to 1 to search all frames)");
                    int GOP = Int32.Parse(Console.ReadLine());
                    Console.WriteLine("Choose a parameter T for (Typically 1-4)");
                    int T = Int32.Parse(Console.ReadLine());
                    Console.WriteLine("Choose a parameter mode (0, 1)");
                    int mode = Int32.Parse(Console.ReadLine());

                    Bitmap[] sData = new Bitmap[4];
                    Bitmap[] Data = new Bitmap[4];
                    Bitmap[] cData = new Bitmap[4];
                    byte[] col = new byte[3];
                    Color _col = new Color();

                    //fv
                    fv = new FeatureVector[frameLength / T];
                    fvp = new FeatureVector[(frameLength / T / GOP)];

                    System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
                    double[] flat = new double[1];
                    //Initialize all feature vectors in fv
                    for(int h = 0; h < fv.Length; h++)
                        fv[h] = new FeatureVector((2*((2*markov_t)+1)*((2*markov_t)+1))+5, -1);
                    //Initialize all feature vectors in fv
                    for (int h = 0; h < fvp.Length; h++)
                        fvp[h] = new FeatureVector((2 * ((2 * markov_t) + 1) * ((2 * markov_t) + 1)) + 5, -1);

                    List<FeatureVector>[] lfv1 = new List<FeatureVector>[k]; // k-cos
                    List<FeatureVector>[] lfv2 = new List<FeatureVector>[k]; // c
                    List<FeatureVector>[] lfv3 = new List<FeatureVector>[k]; // k
                    List<FeatureVector>[] lfv4 = new List<FeatureVector>[k]; // k

                    
                    double[] KurtosisResults = new double[(frameLength / T / GOP)];
                    
                    double[] SkewnessResults = new double[(frameLength / T / GOP)];
                    
                    double[] _1stAbsCResults = new double[(frameLength / T / GOP)];
                    
                    double[] _2ndAbsCResults = new double[(frameLength / T / GOP)];
                    
                    double[] _3rdAbsCResults = new double[(frameLength / T / GOP)];
                    
                    
                    for (int i = 0; i < k; i++)
                    {
                        lfv1[i] = new List<FeatureVector>();
                        lfv2[i] = new List<FeatureVector>();
                        lfv3[i] = new List<FeatureVector>(); 
                        lfv4[i] = new List<FeatureVector>();
                    }
                    double[, ,] DataGroup;
                    for (int ii = -1; ii < (frameLength / T); ii+=(T*GOP))
                    {

                        if (ii == -1)
                        {
                            Console.WriteLine("A. Initializing Steganalytic Tool (8 Steps)");
                            fv[0] = new FeatureVector(1, -1);
                            fv[1] = new FeatureVector(1, -1);
                            //fvp[0] = new FeatureVector(1, -1);
                            //fvp[1] = new FeatureVector(1, -1);
                        }

                        if (ii > -1)
                        {
                            Console.WriteLine("Computing cosine features from blocks: Y(k1, k2, k3, " + ii + ")");
                            //fv[ii + 1] = new FeatureVector(1, ii);
                            fvp[(ii/(T * GOP))] = new FeatureVector(1, ii);

                            for (int i = 0; i < 4; i++)
                                Data[i] = DeepClone(BitmapFromArray1DSafe(GetBytesFromFile(filename, (((videoWidth * videoHeight) * (ii+i)) + (((videoWidth * videoHeight) * (ii+i)) / 2)), videoWidth * videoHeight), videoWidth, videoHeight)); ;
                            
                            DataGroup = M_Group(Data[0], Data[1], Data[2], Data[3], mode);
                            flat = flatten(DataGroup);
                            double mean = m(DataGroup); //Console.WriteLine("Mean of the video is " + mean);
                            double std = H264Stego.Program.std(DataGroup, mean);
                            //Console.WriteLine("Sigma of stegano video is " + std);
                            double Kurtosis = H264Stego.Program.Kurtosis(mean, std, flat);
                            KurtosisResults[ii / (T * GOP)] = Kurtosis;
                            //Console.WriteLine(" Kurtosis: " + Kurtosis);
                            double Skewness = H264Stego.Program.Skewness(mean, std, flat);
                            SkewnessResults[ii / (T * GOP)] = Skewness;
                            //Console.WriteLine(" Skewness: " + Skewness);
                            double s1stabs = AbsCentralMoment(1, videoHeight, videoWidth, flat, mean);
                            _1stAbsCResults[ii / (T * GOP)] = s1stabs;
                            double s2ndabs = AbsCentralMoment(2, videoHeight, videoWidth, flat, mean);
                            _2ndAbsCResults[ii / (T * GOP)] = s2ndabs;
                            double s3rdabs = AbsCentralMoment(3, videoHeight, videoWidth, flat, mean);
                            _3rdAbsCResults[ii / (T * GOP)] = s3rdabs;
                            double blockiness = 0;//S_wsbm(sData[0]);
                            //Console.WriteLine(" 1st, 2nd, 3rd Abs. Central Moments: " + s1stabs + ", " + s2ndabs + ", " + s3rdabs);
                            Markov mv = new Markov();
                            DataGroup = Truncate(DataGroup, mode);
                            mv.markov = transition_Tensor(DataGroup, mode);


                            //Generate a bitmap of the Transition Matrix
                            for (int i = 0; i < mv.markov.GetLength(1); i++)
                                for (int j = 0; j < mv.markov.GetLength(0); j++)
                                {
                                    col = spectrum(mv.markov[j, i, 0]);
                                    _col = Color.FromArgb((col[0] << 16) + (col[1] << 8) + col[2]);
                                    markovian.SetPixel(j, i, _col);
                                }

                            for (int i = 0; i < mv.markov.GetLength(1); i++)
                                markovian.SetPixel((2*markov_t)+1, i, Color.Gray); //Dividing Line

                            for (int i = 0; i < mv.markov.GetLength(1); i++)
                                for (int j = 0; j < mv.markov.GetLength(0); j++)
                                {
                                    col = spectrum(mv.markov[j, i, 1]);
                                    _col = Color.FromArgb((col[0] << 16) + (col[1] << 8) + col[2]);
                                    markovian.SetPixel(j + ((2*markov_t)+1)+1, i, _col);
                                }
                            
                            markovian.Save(Directory.GetCurrentDirectory() + "/graphs/markovian" + ii + ".bmp", ImageFormat.Bmp);

                            //fv[ii + 1] = featureVector1(mv, s1stabs, s2ndabs, s3rdabs, Skewness, Kurtosis, ii, blockiness);
                            fvp[(ii/(T * GOP))] = featureVector1(mv, s1stabs, s2ndabs, s3rdabs, Skewness, Kurtosis, ii, blockiness);

                        }

                        if (ii == -1)
                        {
                            Console.WriteLine("1. Loading Data");
                            for (int i = 0; i < 4; i++)
                                sData[i] = BitmapFromArray1DSafe(GetBytesFromFile(steganovid, (((videoWidth * videoHeight) * i * T) + (((videoWidth * videoHeight) * i * T) / 2)), videoWidth * videoHeight), videoWidth, videoHeight); ;

                            for (int i = 0; i < 4; i++)
                                cData[i] = BitmapFromArray1DSafe(GetBytesFromFile(cleanvid, (((videoWidth * videoHeight) * i * T) + (((videoWidth * videoHeight) * i * T) / 2)), videoWidth * videoHeight), videoWidth, videoHeight); ;

                            Console.WriteLine("2. Performing 3D DCT, Acquiring Groups Y(k1,k2,k3,k) for Initial Centroids, calculating Low-Order Statistics");
                            double[, ,] sDataGroup = M_Group(sData[0], sData[1], sData[2], sData[3], mode);
                            double[, ,] cDataGroup = M_Group(cData[0], cData[1], cData[2], cData[3], mode);

                            Console.WriteLine("2. Assembling transform histograms");

                            double smean = m(sDataGroup); Console.WriteLine("Mean of stegano video is" + smean);
                            double cmean = m(cDataGroup); Console.WriteLine("Mean of clean video is " + cmean);


                            double sstd = std(sDataGroup, smean); Console.WriteLine("Sigma of stegano video is " + sstd);
                            double cstd = std(cDataGroup, cmean); Console.WriteLine("Sigma of clean video is " + cstd);


                            Console.WriteLine("\n3. Calculating Kurtosis for Initial Centroids");
                            double sKurtosis = Kurtosis(smean, sstd, flatten(sDataGroup));
                            double cKurtosis = Kurtosis(cmean, cstd, flatten(cDataGroup));
                            Console.WriteLine("Kurtosis for Clean Videos: " + cKurtosis);
                            Console.WriteLine("Kurtosis for Dirty Videos: " + sKurtosis);

                            Console.WriteLine("\n4. Calculating Skewness for Initial Centroids");
                            double sSkewness = Skewness(smean, sstd, flatten(sDataGroup));
                            double cSkewness = Skewness(cmean, cstd, flatten(cDataGroup));
                            Console.WriteLine("Skewness for Clean Videos: " + cSkewness);
                            Console.WriteLine("Skewness for Dirty Videos: " + sSkewness);

                            Console.WriteLine("\n5. Calculating 1st order Absolute Central Moment for Initial Centroids");
                            double s1stabs = AbsCentralMoment(1, videoHeight, videoWidth, flatten(sDataGroup), smean);
                            double c1stabs = AbsCentralMoment(1, videoHeight, videoWidth, flatten(cDataGroup), cmean);
                            Console.WriteLine("1st order Absolute Moment for Clean Videos: " + c1stabs);
                            Console.WriteLine("1st order Absolute Moment for Dirty Videos: " + s1stabs);

                            Console.WriteLine("\n6. Calculating 2nd order Absolute Central Moment for Initial Centroids");
                            double s2ndabs = AbsCentralMoment(2, videoHeight, videoWidth, flatten(sDataGroup), smean);
                            double c2ndabs = AbsCentralMoment(2, videoHeight, videoWidth, flatten(cDataGroup), cmean);
                            Console.WriteLine("2nd order Absolute Moment for Clean Videos: " + c2ndabs);
                            Console.WriteLine("2nd order Absolute Moment for Dirty Videos: " + s2ndabs);

                            Console.WriteLine("\n7. Calculating 3rd order Absolute Central Moment for Initial Centroids");
                            double s3rdabs = AbsCentralMoment(3, videoHeight, videoWidth, flatten(sDataGroup), smean);
                            double c3rdabs = AbsCentralMoment(3, videoHeight, videoWidth, flatten(cDataGroup), cmean);
                            Console.WriteLine("3rd order Absolute Moment for Clean Videos: " + c3rdabs);
                            Console.WriteLine("3rd order Absolute Moment for Dirty Videos: " + s3rdabs);

                            Console.WriteLine("\n8. Calculating Markov Features for Initial Centroids");
                            Markov smv = new Markov();
                            Markov cmv = new Markov();

                            
                            double sblockiness = S_wsbm(sData[0]);
                            double cblockiness = S_wsbm(cData[0]);

                            Console.WriteLine("Blockiness for the stegano video " + sblockiness);
                            Console.WriteLine("Blockiness for the clean video   " + cblockiness);

                            //Reduce feature dimension by T, which is 10
                            sDataGroup = Truncate(sDataGroup, mode);
                            cDataGroup = Truncate(cDataGroup, mode);

                            smv.markov = transition_Tensor(sDataGroup, mode);
                            cmv.markov = transition_Tensor(cDataGroup, mode);

                            Console.WriteLine("\n8. Assembling essential feature Vectors for initializing 2-means");
                            fv[0] = featureVector1(smv, s1stabs, s2ndabs, s3rdabs, sSkewness, sKurtosis, -1, 0);
                            fv[1] = featureVector1(cmv, c1stabs, c2ndabs, c3rdabs, cSkewness, cKurtosis, -1, 0);
                            //fvp[0] = featureVector1(smv, s1stabs, s2ndabs, s3rdabs, sSkewness, sKurtosis, -1, sblockiness);
                            //fvp[1] = featureVector1(cmv, c1stabs, c2ndabs, c3rdabs, cSkewness, cKurtosis, -1, cblockiness);

                            Console.WriteLine("Centroid 0 is Stegano");
                            Console.WriteLine("Centroid 1 is Clean");
                            if (ii == -1)
                            {
                                Console.WriteLine("Done Initializing! \n\nB. Gathering Low-order and High-order statistics from target video frames.");
                                ii = (-T * GOP);
                            }
                        }



                        //ii = 295;
                    }

                    FeatureVector[] f_ = new FeatureVector[2+(frameLength/GOP)];
/*
                    for (int i = 0; i < k; i++)
                        f_[i] = DeepClone(fv[i]);

                    for (int i = 2; i < (frameLength/GOP)+2; i++)
                        f_[i] = DeepClone(fv[(((i-2) * GOP)+1)]);
    */                

                    
                    //fv = new FeatureVector[fvp.Length];
                    //fv = DeepClone(fvp);

                    for (int i = 0; i < k; i++)
                        lfv1[i].Add(DeepClone(fv[i]));

                    for (int i = 0; i < k; i++)
                        lfv2[i].Add(DeepClone(fv[i]));

                    for (int i = 0; i < k; i++)
                        lfv3[i].Add(DeepClone(fv[i]));

                    for (int i = 0; i < k; i++)
                        lfv4[i].Add(DeepClone(fv[i]));

                    FeatureVector[] fv2 = new FeatureVector[fvp.Length];
                    FeatureVector[] fv3 = new FeatureVector[fvp.Length];
                    FeatureVector[] fv4 = new FeatureVector[fvp.Length];
                    fv2 = DeepClone(fvp);
                    fv3 = DeepClone(fvp);
                    fv4 = DeepClone(fvp);

                    Console.WriteLine("Kurtosis For Frames:");
                    foreach (double d in KurtosisResults)
                        Console.WriteLine(d);
                    Console.ReadLine();
                    Console.WriteLine("Skewness For Frames:");
                    foreach (double d in SkewnessResults)
                        Console.WriteLine(d);
                    Console.ReadLine();
                    Console.WriteLine("1st Absolute Central Moment For Frames:");
                    foreach (double d in _1stAbsCResults)
                        Console.WriteLine(d);
                    Console.ReadLine();
                    Console.WriteLine("2nd Absolute Central Moment For Frames:");
                    foreach (double d in _2ndAbsCResults)
                        Console.WriteLine(d);
                    Console.ReadLine();
                    Console.WriteLine("3rd Absolute Central Moment For Frames:");
                    foreach (double d in _3rdAbsCResults)
                        Console.WriteLine(d);
                    Console.ReadLine();
                    Console.WriteLine('\n' + '\n');

                    Console.WriteLine("\nSPHERICAL K-MEANS:");
                    sw = System.Diagnostics.Stopwatch.StartNew();
                    lfv1 = k_means_cosine(fvp, lfv1, k, 9999);
                    sw.Stop();
                    Console.WriteLine("Time taken in SKM is " + sw.Elapsed.TotalMilliseconds);

                    Console.WriteLine("\nFUZZY C-MEANS:");
                    sw = System.Diagnostics.Stopwatch.StartNew();
                    fuzzy_c_means(fv2, k, 0.1, new FeatureVector[] {DeepClone(fv[0]), DeepClone(fv[1])});
                    sw.Stop();
                    Console.WriteLine("Time taken in FCM is " + sw.Elapsed.TotalMilliseconds);

                    Console.WriteLine("\nSPHERICAL FUZZY C-MEANS:");
                    sw = System.Diagnostics.Stopwatch.StartNew();
                    spherical_fuzzy_c_means(fv3, k, 0.1, new FeatureVector[] { DeepClone(fv[0]), DeepClone(fv[1]) });
                    sw.Stop();
                    Console.WriteLine("Time taken in FCM is " + sw.Elapsed.TotalMilliseconds);

                    Console.WriteLine("\nK-MEANS:");
                    sw = System.Diagnostics.Stopwatch.StartNew();
                    lfv4 = k_means(fv4, lfv4, k, 9999);
                    sw.Stop();
                    Console.WriteLine("Time taken in K-MEANS is " + sw.Elapsed.TotalMilliseconds);


                    Console.ReadLine();
                }


                for (int i = 0; i < frameLength - 3; i++)
                {


                    Console.Write("\n Scanning the video with " + (100 * (((double)i / (double)frameLength))) + "% Completed \n");

                    videoData = GetBytesFromFile(filename, (((videoWidth * videoHeight) * i) + (((videoWidth * videoHeight) * i) / 2)), videoWidth * videoHeight);
                    b = BitmapFromArray1D(videoData, videoWidth, videoHeight);

                    videoData = GetBytesFromFile(filename, (((videoWidth * videoHeight) * (i + 1)) + (((videoWidth * videoHeight) * (i + 1)) / 2)), videoWidth * videoHeight);
                    b1 = BitmapFromArray1D(videoData, videoWidth, videoHeight);

                    videoData = GetBytesFromFile(filename, (((videoWidth * videoHeight) * (i + 2)) + (((videoWidth * videoHeight) * (i + 2)) / 2)), videoWidth * videoHeight);
                    b2 = BitmapFromArray1D(videoData, videoWidth, videoHeight);

                    videoData = GetBytesFromFile(filename, (((videoWidth * videoHeight) * (i + 3)) + (((videoWidth * videoHeight) * (i + 3)) / 2)), videoWidth * videoHeight);
                    b3 = BitmapFromArray1D(videoData, videoWidth, videoHeight);

                    steg = DCT3D_4x4x4(b, b1, b2, b3, 0);

                    videoData = Array1DFromBitmap(b, videoWidth * videoHeight);
                    SaveBytesToFile(filename, (((videoWidth * videoHeight) * i) + (((videoWidth * videoHeight) * i) / 2)), videoWidth * videoHeight, videoData);

                }

            }

            if (hiding == "R")
            {
                Console.WriteLine("Enter the filename to save the reconstructed file to");
                savefilename = Console.ReadLine();
                Console.WriteLine("What is the filesize of the file in bytes? If unknown, type 42");
                filesizespecified = long.Parse(Console.ReadLine());
            }

            Console.WriteLine("What algorithm will you use? <Choose Number> \n \n \n 1. (checkers): Spatial Domain PVD - Very Robust, uses computationally efficient Pixel Value Differencing and Histogram Shifting algorithms. Easily detectable visually however. Payload capacity is large. Reccomended if video is being uploaded to YouTube or highly compressed on disk. Not recommended for covert operation and is weak to known steganalysis attacks.");
            Console.WriteLine("\n \n \n 2. (smiling face): Transform Domain DCT2D 8-D Vector Quantization - Fragile, with borderline robustness, uses computationally intensive Discrete Cosine Transform and eight dimensional vector quantization. Invisible with distortion < 3db (6% visual distortion calculated with PSNR). Payload capacity is large but small once error correction used. Highly recommended if video is stored on hard disk and will not be re-encoded. Recommended for covert operation as embedded data is invisible.");
            string response = Console.ReadLine();
            Console.WriteLine("What is GOP?");
            int GOP_ = Int32.Parse(Console.ReadLine());
            while (true)
            {
                if (response == "2")
                {
                    Console.WriteLine("What is your thresholding parameter T?");
                    int T = Int32.Parse(Console.ReadLine());
                    Console.WriteLine("What is the error correcting codeword length you'll use? (Set this to 900 or greater for YouTube.)");
                    int rep = Int32.Parse(Console.ReadLine());
                    bytes = GetBytesFromFile(sfilename, 0);
                    acc = new double[8 * bytes.Length];

                    finalFile = new byte[bytes.Length];

                    if (bytes.Length * 8 * rep > videoWidth * videoHeight / 4 / 4)
                    {
                        Console.WriteLine(" * File is much larger than embedding capacity of frame with chosen error code, would you like to embed it across all frames? <Y/n>");
                        string allframes = Console.ReadLine();

                        if (allframes != "Y")
                        {
                            Console.WriteLine("File too large, aborting...");
                        }
                        else
                        {
                            loop = false;
                        }
                    }
                    Frequency_4x4 f = new Frequency_4x4();



                    for (int i = 0; i < frameLength; i+= GOP_)
                    {
                        if (hiding == "H")
                        {

                            Console.Write("\n Processing.." + (100 * (((double)i/(double)frameLength))) + "% Completed \n");

                            videoData = GetBytesFromFile(filename, (((videoWidth * videoHeight) * i) + (((videoWidth * videoHeight) * i) / 2)), videoWidth * videoHeight);
                            b = BitmapFromArray1DSafe(videoData, videoWidth, videoHeight);
                            b = transform_Embed(b, T, bytes, rep);

                            videoData = Array1DFromBitmapSafe(b, videoWidth * videoHeight);
                            SaveBytesToFile(filename, (((videoWidth * videoHeight) * i) + (((videoWidth * videoHeight) * i) / 2)), videoWidth * videoHeight, videoData);

                        }
                        if (hiding == "R")
                        {

                            if (bytes.Length * rep > (videoWidth * videoHeight / 8 / 16))
                                loop = false;

                            filesizespecified = bytes.Length;

                            Console.Write("\n Processing.." + (100 * (((double)i / (double)frameLength))) + "% Completed \n");

                            videoData = GetBytesFromFile(filename, (((videoWidth * videoHeight) * i) + (((videoWidth * videoHeight) * i) / 2)), videoWidth * videoHeight);
                            b = BitmapFromArray1DSafe(videoData, videoWidth, videoHeight);
                            saveFile = transform_Retrieve(b, bytes, T, rep);

                            if (!loop)
                                Console.WriteLine("BER for chunk " + i + " (Bit Error Rate) is " + BER(bytes, saveFile));//, (int)(pos), (int)(videoWidth * videoHeight / 16 / rep)));
                            if (loop)
                                Console.WriteLine("BER (Bit Error Rate) is " + BER(saveFile, bytes));

                            SaveBytesToFile(savefilename, 0, bytes.Length, saveFile);
                            if (loop)
                                acc = AccArray(saveFile, acc);

                            Console.ReadLine();
                        }

                    }
                }
                if (response == "1")
                {
                    Console.WriteLine("What is your thresholding parameter T?");
                    int T = Int32.Parse(Console.ReadLine());
                    Console.WriteLine("What is your thresholding parameter G?");
                    int G = Int32.Parse(Console.ReadLine());
                    Console.WriteLine("What is the block size");
                    int block_size = Int32.Parse(Console.ReadLine());
                    Console.WriteLine("What is the spatial error correcting codeword length you'll use? (Set this to 50 or greater for YouTube.)");
                    int rep = Int32.Parse(Console.ReadLine());
                    Console.WriteLine("What is the temporal error correcting codeword length you'll use? (Set this to 50 or greater for YouTube.)");
                    int trep = Int32.Parse(Console.ReadLine());
                    bytes = GetBytesFromFile(sfilename, 0);
                    finalFile = new byte[bytes.Length];
                    acc = new double[8 * bytes.Length];
                    
                    long loc = 0;
                    if (bytes.Length * 8 * rep > videoWidth * videoHeight / block_size / block_size)
                    {
                        Console.WriteLine(" * File is much larger than embedding capacity of frame with chosen error code, would you like to embed it across all frames? <Y/n>");
                        string allframes = Console.ReadLine();

                        if (allframes != "Y")
                        {
                            Console.WriteLine("File too large, aborting...");

                        }

                        else
                        {
                            loop = false;
                        }

                    }

                    for (int i = 0; i < frameLength; i+=GOP_)
                    {

                        if (hiding == "H")
                        {

                            Console.Write("\n Processing.." + (100 * (((double)i / (double)frameLength))) + "% Completed \n");

                            videoData = GetBytesFromFile(filename, (((videoWidth * videoHeight) * i) + (((videoWidth * videoHeight) * i) / 2)), videoWidth * videoHeight);
                            b = BitmapFromArray1D(videoData, videoWidth, videoHeight);
                            b = spatial_Embed(b, block_size, T, G, bytes, rep, trep);

                            videoData = Array1DFromBitmap(b, videoWidth * videoHeight);
                            SaveBytesToFile(filename, (((videoWidth * videoHeight) * i) + (((videoWidth * videoHeight) * i) / 2)), videoWidth * videoHeight, videoData);

                        }
                        if (hiding == "R")
                        {

                            if (bytes.Length * rep > (videoWidth * videoHeight / 8 / block_size / block_size))
                                loop = false;

                            filesizespecified = bytes.Length;


                            Console.Write("\n Processing.." + (100 * (((double)i / (double)frameLength))) + "% Completed.");
                            loc = pos;
                            videoData = GetBytesFromFile(filename, (((videoWidth * videoHeight) * i) + (((videoWidth * videoHeight) * i) / 2)), videoWidth * videoHeight);
                            b = BitmapFromArray1D(videoData, videoWidth, videoHeight);
                            saveFile = spatial_Retrieve(b, block_size, T, G, bytes, rep, trep);

                            if (!loop)
                                if ((i + 1) % (trep) == 0)
                                {
                                    Console.WriteLine("BER for chunk " + i + " (Bit Error Rate) is " + BER(bytes, saveFile));//, (int)(loc), (int)(pos - loc)));
                                    // I remember what this does, we don't need it right now. Array.Copy(saveFile, pos, finalFile, pos, pos - loc); //this is the cumulative file
                                    Console.WriteLine((100 * (((double)i / (double)frameLength))) + " % of the file ( " + pos + " ) bytes : total filesize of ( " + bytes.Length + " ) bytes has been saved to " + savefilename);
                                    SaveBytesToFile(savefilename, 0, bytes.Length, saveFile); //save the whole file
                                }

                            if (loop)
                                Console.WriteLine("BER (Bit Error Rate) is " + BER(bytes, saveFile));
                            if (loop)
                                SaveBytesToFile(savefilename, 0, bytes.Length, saveFile); //save this
                            if (loop)
                                acc = AccArray(saveFile, acc);

                            Console.ReadLine();
                        }

                    }
                    acc = AvgArray(acc, frameLength);
                    Console.WriteLine("P value is " + P(acc));
                    Console.WriteLine("SUCCESS! Completed 100%. Please be careful with this video.");
                    Console.ReadLine();
                }
                Console.WriteLine("Try a different response.");
                Console.ReadLine();
            }
        }
    }
}

/*
            Cited

        [1] : Hong Zhao; Hongxia Wang; Malik, H., 
 * 
 * 
 * "Steganalysis of Youtube Compressed Video Using High-order Statistics in 3D DCT Domain," 
 *      in Intelligent Information Hiding and Multimedia Signal Processing (IIH-MSP), 
 *      2012 Eighth International Conference on , vol., no., pp.191-194, 18-20 July 2012
        doi: 10.1109/IIH-MSP.2012.52
 
 * 
 * 
 * 
 * keywords: {
 * Markov processes;
 * discrete cosine transforms;
 * image classification;
 * pattern clustering;
 * social networking (online);
 * spatiotemporal phenomena;
 * statistical analysis;
 * steganography;
 * video coding;
 * 3D DCT domain;
 * 3D discrete cosine transform domain;
 * Markov features;
 * YouTube compressed video steganalysis;
 * absolute central moments;
 * cover-videos;
 * data hiding;
 * hidden message detection;
 * high-order statistics classification;
 * kurtosis;
 * skewness;
 * spatial-temporal correlation;
 * stego-videos;
 * unsupervised k-means clustering;
 * video frames;
 * Correlation;
 * Discrete cosine transforms;
 * Feature extraction;
 * Image coding;
 * Markov processes;
 * Video compression;
 * YouTube;
 * 3D-DCT;
 * Video Steganalysis},

 
 */
