// Copyright 2026 Aljoscha Schmidt
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

#include <graph/NodesCore.hpp>

void StereoOcclusionPointsNode::setParameters(int32_t width, float focalLength, float baseline, const Vec3f& opticalAxis,
                                              int32_t matchingBand)
{
	this->width = width;
	this->focalLength = focalLength;
	this->baseline = baseline;
	this->opticalAxis = opticalAxis.normalized();
	this->matchingBand = matchingBand;
}

void StereoOcclusionPointsNode::enqueueExecImpl()
{
	auto pointCount = input->getPointCount();
	if (pointCount % width != 0) {
		auto msg = fmt::format("{}: {} points do not fill rows of {} pixels", getName(), pointCount, width);
		throw InvalidPipeline(msg);
	}
	unseen->resize(pointCount, false, false);
	outXyz->resize(pointCount, false, false);
	outDistance->resize(pointCount, false, false);
	outIsHit->resize(pointCount, false, false);

	const auto* inXyzPtr = input->getFieldDataTyped<XYZ_VEC3_F32>()->asSubclass<DeviceAsyncArray>()->getReadPtr();
	const auto* inDistancePtr = input->getFieldDataTyped<DISTANCE_F32>()->asSubclass<DeviceAsyncArray>()->getReadPtr();
	const auto* inIsHitPtr = input->getFieldDataTyped<IS_HIT_I32>()->asSubclass<DeviceAsyncArray>()->getReadPtr();
	gpuFindStereoUnseenPoints(getStreamHandle(), pointCount / width, width, focalLength * fabsf(baseline), baseline > 0.0f,
	                          opticalAxis, input->getLookAtOriginTransform(), inXyzPtr, inIsHitPtr, unseen->getWritePtr());
	gpuRemoveStereoUnseenPoints(getStreamHandle(), pointCount, width, matchingBand, unseen->getReadPtr(), inXyzPtr,
	                            inDistancePtr, inIsHitPtr, outXyz->getWritePtr(), outDistance->getWritePtr(),
	                            outIsHit->getWritePtr());
}

IAnyArray::ConstPtr StereoOcclusionPointsNode::getFieldData(rgl_field_t field)
{
	if (field == XYZ_VEC3_F32) {
		return outXyz;
	}
	if (field == DISTANCE_F32) {
		return outDistance;
	}
	if (field == IS_HIT_I32) {
		return outIsHit;
	}
	return input->getFieldData(field);
}
