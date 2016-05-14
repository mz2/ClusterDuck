//
//  MPGaussianMixtureModel.m
//  ClusterDuck
//
//  Created by Matias Piipari on 14/05/2016.
//  Copyright © 2016 Matias Piipari & Co. All rights reserved.
//

#import "MPGaussianMixtureModel.h"

#import "libcluster.h"
#import "distributions.h"
#import "testdata.h"

#import <vector>

using namespace std;
using namespace Eigen;
using namespace libcluster;
using namespace distributions;

@interface MPBayesianGaussianMixture ()
- (instancetype)initWithExpectedClusterAssignments:(NSArray <NSNumber *> *)assignments
                                    clusterWeights:(NSArray <NSNumber *> *)weights
                                clusterCovariances:(NSArray <NSNumber *> *)covariances;
@end

@implementation MPGaussianMixtureModel

+ (std::vector<Eigen::MatrixXd>)doubleMatricesForInput:(NSArray<NSArray<NSNumber *> *> *)numbers {
    NSUInteger rowCount = numbers.count;
    NSUInteger colCount = numbers.firstObject.count;
    NSParameterAssert(rowCount > 0);
    NSParameterAssert(colCount > 0);
    
    std::vector<Eigen::MatrixXd> X;
    
    for (NSUInteger r = 0; r < rowCount;r++) {
        NSArray<NSNumber *> *row = numbers[r];
        NSAssert(row.count == colCount, @"Unexpected number of columns at row index %@: %@", @(r), row);
        
        Eigen::MatrixXd rowMatrix = Eigen::MatrixXd(1, colCount);
        for (NSUInteger c = 0; c < colCount; c++) {
            rowMatrix(0, c) = [row[c] doubleValue];
        }
        
        X.push_back(rowMatrix);
    }
    
    return X;
}

+ (Eigen::MatrixXd)doubleMatrixForInput:(NSArray<NSArray<NSNumber *> *> *)numbers {
    NSUInteger rowCount = numbers.count;
    NSUInteger colCount = numbers.firstObject.count;
    
    Eigen::MatrixXd M(rowCount, colCount);
    
    for (NSUInteger r = 0; r < rowCount;r++) {
        NSArray<NSNumber *> *row = numbers[r];
        NSAssert(row.count == colCount, @"Unexpected number of columns at row index %@: %@", @(r), row);
        
        for (NSUInteger c = 0; c < colCount; c++) {
            M(r, c) = [row[c] doubleValue];
        }
    }
    
    return M;
}

- (NSArray<NSNumber *> *)expectedClusterAssignmentsForPosteriorDistribution:(const MatrixXd &)qZ {
    NSMutableArray *expectedClusters = [NSMutableArray new];
    
    for (long r = 0, rows = qZ.rows(); r < rows; r++) {
        long maxIndex = -1;
        double max = 0;
        for (long c = 0, cols = qZ.cols(); c < cols; c++) {
            double v = qZ(r, c);
            
            if (max < v) {
                max = v;
                maxIndex = c;
            }
        }
        
        NSParameterAssert(maxIndex >= 0);
        [expectedClusters addObject:@(maxIndex)];
    }
    
    return expectedClusters;
}

+ (NSArray<NSNumber *> *)bayesianGaussianMixtureModelForInput:(NSArray<NSArray<NSNumber *> *> *)numbers {
    MatrixXd X = [self.class doubleMatrixForInput:numbers];
    
    //vector<GDirichlet> weights;
    vector<GaussWish>  clusters;
    
    distributions::Dirichlet weights;
    MatrixXd qZ;
    
    clock_t start = clock();
    double freeEnergy = learnBGMM(X, qZ, weights, clusters, 1);
    double stop = (double)((clock() - start))/CLOCKS_PER_SEC;
    
    cout << "BMCC Elapsed time = " << stop << " sec." << endl;
    
    cout << endl << "Remaining free energy: " << freeEnergy << endl;
    
    cout << endl << "Cluster Weights:" << weights.Elogweight().exp().transpose() << endl;
    
    cout << endl << "Variational Posterior :" << weights.Elogweight().exp().transpose() << endl;
    for (long r = 0, rows = qZ.rows(); r < rows; r++) {
        for (long c = 0, cols = qZ.cols(); c < cols; c++) {
            cout << qZ(r, c);
            
            if (c < (qZ.cols() - 1)) {
                cout << ",";
            }
        }
        cout << endl;
    }
    
    cout << "Cluster means" << endl;
    for (vector<GaussWish>::iterator k=clusters.begin(); k < clusters.end(); ++k)
        cout << k->getmean() << endl;
    
    cout << endl << "Cluster covariances:" << endl;
    for (vector<GaussWish>::iterator k=clusters.begin(); k < clusters.end(); ++k)
        cout << k->getcov() << endl << endl;
    
    return nil;
}

@end


@implementation MPBayesianGaussianMixture

- (instancetype)initWithExpectedClusterAssignments:(NSArray <NSNumber *> *)assignments
                                    clusterWeights:(NSArray <NSNumber *> *)weights
                                clusterCovariances:(NSArray <NSNumber *> *)covariances {
    self = [super init];
    
    if (self) {
        _expectedClusterAssignments = assignments;
        _clusterWeights = weights;
        _clusterCovariances = covariances;
    }
    
    return self;
}

@end
