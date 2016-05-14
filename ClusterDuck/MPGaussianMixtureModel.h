//
//  MPGaussianMixtureModel.h
//  ClusterDuck
//
//  Created by Matias Piipari on 14/05/2016.
//  Copyright © 2016 Matias Piipari & Co. All rights reserved.
//

#import <Foundation/Foundation.h>

@interface MPGaussianMixtureModel : NSObject

- (instancetype)init NS_UNAVAILABLE;

+ (NSArray<NSNumber *> *)bayesianGaussianMixtureModelForInput:(NSArray<NSArray<NSNumber *> *> *)numbers;

@end
