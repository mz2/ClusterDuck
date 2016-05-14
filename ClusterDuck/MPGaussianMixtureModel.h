//
//  MPGaussianMixtureModel.h
//  ClusterDuck
//
//  Created by Matias Piipari on 14/05/2016.
//  Copyright © 2016 Matias Piipari & Co. All rights reserved.
//

#import <Foundation/Foundation.h>

@class MPBayesianGaussianMixture;

@interface MPGaussianMixtureModel : NSObject

- (nonnull instancetype)init NS_UNAVAILABLE;

+ (nonnull NSArray<MPBayesianGaussianMixture*> *)bayesianGaussianMixtureModelForInput:(nonnull NSArray<NSArray<NSNumber *> *> *)numbers;

@end

@interface MPBayesianGaussianMixture: NSObject

- (nonnull instancetype)init NS_UNAVAILABLE;

@property (readonly, nonnull) NSArray <NSNumber *> *expectedClusterAssignments;
@property (readonly, nonnull) NSArray <NSNumber *> *clusterWeights;
@property (readonly, nonnull) NSArray <NSNumber *> *clusterCovariances;

@end
