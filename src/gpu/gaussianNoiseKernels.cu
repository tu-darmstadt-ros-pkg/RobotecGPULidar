// Copyright 2023 Robotec.AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cuda.h>
#include <curand_kernel.h>
#include <gpu/kernelUtils.hpp>
#include <gpu/gaussianNoiseKernels.hpp>

__global__ void kAddGaussianNoiseAngularRay(size_t rayCount, float mean, float stDev, rgl_axis_t rotationAxis,
                                            Mat3x4f lookAtOriginTransform, curandStatePhilox4_32_10_t* randomStates,
                                            const Mat3x4f* inRays, Mat3x4f* outRays)
{
	LIMIT(rayCount);

	float angularError = mean + curand_normal(&randomStates[tid]) * stDev;
	outRays[tid] = lookAtOriginTransform.inverse() *
	               (Mat3x4f::rotationRad(rotationAxis, angularError) * (lookAtOriginTransform * inRays[tid]));
}

__global__ void kAddGaussianNoiseAngularHitpoint(size_t pointCount, float mean, float stDev, rgl_axis_t rotationAxis,
                                                 Mat3x4f lookAtOriginTransform, curandStatePhilox4_32_10_t* randomStates,
                                                 const Field<XYZ_VEC3_F32>::type* inPoints,
                                                 Field<XYZ_VEC3_F32>::type* outPoints, Field<DISTANCE_F32>::type* outDistances)
{
	LIMIT(pointCount);

	float angularError = mean + curand_normal(&randomStates[tid]) * stDev;
	Field<XYZ_VEC3_F32>::type originWithNoisePoint = Mat3x4f::rotationRad(rotationAxis, angularError) *
	                                                 (lookAtOriginTransform * inPoints[tid]);

	if (outDistances != nullptr) {
		outDistances[tid] = originWithNoisePoint.length();
	}

	outPoints[tid] = lookAtOriginTransform.inverse() * originWithNoisePoint;
}

__global__ void kAddGaussianNoiseDistance(size_t pointCount, float mean, float stDevBase, float stDevRisePerMeter,
                                          float stDevRisePerMeterSquared, float maxIncidenceAngle,
                                          Mat3x4f lookAtOriginTransform, curandStatePhilox4_32_10_t* randomStates,
                                          const Field<XYZ_VEC3_F32>::type* inPoints,
                                          const Field<DISTANCE_F32>::type* inDistances, const Field<IS_HIT_I32>::type* inIsHit,
                                          const Field<INCIDENT_ANGLE_F32>::type* inIncidentAngles,
                                          Field<XYZ_VEC3_F32>::type* outPoints, Field<DISTANCE_F32>::type* outDistances,
                                          Field<IS_HIT_I32>::type* outIsHit)
{
	LIMIT(pointCount);

	outIsHit[tid] = inIsHit[tid];
	if (!inIsHit[tid]) {
		outPoints[tid] = inPoints[tid];
		outDistances[tid] = inDistances[tid];
		return;
	}

	float cosIncidentAngle = 1.0f;
	if (inIncidentAngles != nullptr) {
		if (inIncidentAngles[tid] > maxIncidenceAngle) {
			outIsHit[tid] = 0;
			outPoints[tid] = Vec3f{NAN, NAN, NAN};
			outDistances[tid] = NAN;
			return;
		}
		cosIncidentAngle = cosf(inIncidentAngles[tid]);
	}

	const float distance = inDistances[tid];
	const float totalStDev = (stDevBase + stDevRisePerMeter * distance + stDevRisePerMeterSquared * distance * distance) /
	                         cosIncidentAngle;
	const float distanceError = mean + curand_normal(&randomStates[tid]) * totalStDev;

	Field<XYZ_VEC3_F32>::type pointInRayOriginTransform = lookAtOriginTransform * inPoints[tid];

	outPoints[tid] = lookAtOriginTransform.inverse() *
	                 (pointInRayOriginTransform + pointInRayOriginTransform.normalized() * distanceError);
	outDistances[tid] = distance + distanceError;
}

__global__ void kAddGaussianNoiseRayDirection(size_t rayCount, float stDev, curandStatePhilox4_32_10_t* randomStates,
                                              const Mat3x4f* inRays, Mat3x4f* outRays)
{
	LIMIT(rayCount);

	// A ray points along its z; it tilts about its x and y.
	const float2 tilt = curand_normal2(&randomStates[tid]);
	outRays[tid] = inRays[tid] * Mat3x4f::rotationRad(tilt.x * stDev, tilt.y * stDev, 0.0f);
}

void gpuAddGaussianNoiseAngularRay(cudaStream_t stream, size_t rayCount, float mean, float stDev, rgl_axis_t rotationAxis,
                                   Mat3x4f lookAtOriginTransform, curandStatePhilox4_32_10_t* randomStates,
                                   const Mat3x4f* inRays, Mat3x4f* outRays)
{
	run(kAddGaussianNoiseAngularRay, stream, rayCount, mean, stDev, rotationAxis, lookAtOriginTransform, randomStates, inRays,
	    outRays);
}

void gpuAddGaussianNoiseAngularHitpoint(cudaStream_t stream, size_t pointCount, float mean, float stDev,
                                        rgl_axis_t rotationAxis, Mat3x4f lookAtOriginTransform,
                                        curandStatePhilox4_32_10_t* randomStates, const Field<XYZ_VEC3_F32>::type* inPoints,
                                        Field<XYZ_VEC3_F32>::type* outPoints, Field<DISTANCE_F32>::type* outDistances)
{
	run(kAddGaussianNoiseAngularHitpoint, stream, pointCount, mean, stDev, rotationAxis, lookAtOriginTransform, randomStates,
	    inPoints, outPoints, outDistances);
}

void gpuAddGaussianNoiseDistance(cudaStream_t stream, size_t pointCount, float mean, float stDevBase, float stDevRisePerMeter,
                                 float stDevRisePerMeterSquared, float maxIncidenceAngle, Mat3x4f lookAtOriginTransform,
                                 curandStatePhilox4_32_10_t* randomStates, const Field<XYZ_VEC3_F32>::type* inPoints,
                                 const Field<DISTANCE_F32>::type* inDistances, const Field<IS_HIT_I32>::type* inIsHit,
                                 const Field<INCIDENT_ANGLE_F32>::type* inIncidentAngles, Field<XYZ_VEC3_F32>::type* outPoints,
                                 Field<DISTANCE_F32>::type* outDistances, Field<IS_HIT_I32>::type* outIsHit)
{
	run(kAddGaussianNoiseDistance, stream, pointCount, mean, stDevBase, stDevRisePerMeter, stDevRisePerMeterSquared,
	    maxIncidenceAngle, lookAtOriginTransform, randomStates, inPoints, inDistances, inIsHit, inIncidentAngles, outPoints,
	    outDistances, outIsHit);
}

void gpuAddGaussianNoiseRayDirection(cudaStream_t stream, size_t rayCount, float stDev,
                                     curandStatePhilox4_32_10_t* randomStates, const Mat3x4f* inRays, Mat3x4f* outRays)
{
	run(kAddGaussianNoiseRayDirection, stream, rayCount, stDev, randomStates, inRays, outRays);
}
