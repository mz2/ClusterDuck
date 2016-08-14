//
//  MPGaussianMixtureModel.h
//  ClusterDuck
//
//  Created by Matias Piipari on 14/05/2016.
//  Copyright © 2016 Matias Piipari & Co. All rights reserved.
//

#import <Foundation/Foundation.h>

@class MPBayesianGaussianMixture;

@interface MPSummaryBGMM

@property (readonly) double freeEnergy;
@property (readonly) double clusterWeights;
@property (readonly, nonnull) NSArray<NSNumber *> *covariances;

@end

@interface MPGaussianMixtureModel : NSObject

- (nonnull instancetype)init NS_UNAVAILABLE;

+ (nonnull NSArray<NSNumber *> *)bayesianGaussianMixtureModelForInput:(nonnull NSArray<NSArray<NSNumber *> *> *)numbers
                                                  dirichletPriorAlpha:(double)alpha
                                                 expectedClusterCount:(NSUInteger)expectedComponents;

@end

@interface MPBayesianGaussianMixture: NSObject

- (nonnull instancetype)init NS_UNAVAILABLE;

@property (readonly, nonnull) NSArray <NSNumber *> *expectedClusterAssignments;
@property (readonly, nonnull) NSArray <NSNumber *> *clusterWeights;
@property (readonly, nonnull) NSArray <NSNumber *> *clusterCovariances;

@end
