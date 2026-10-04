#include <helpers/commonHelpers.hpp>
#include <helpers/mathHelpers.hpp>
#include <helpers/sceneHelpers.hpp>

#include <RGLFields.hpp>
#include <math/Mat3x4f.hpp>

class GaussianNoiseDistanceNodeTest : public RGLTest
{
protected:
	static constexpr int RAY_COUNT = 100000;
	static constexpr float EPSILON_NOISE = 0.002f;
	static constexpr float NON_HIT_DISTANCE = 42.0f;
	static constexpr float ANGLE_INDEPENDENT = static_cast<float>(M_PI_2);

	rgl_node_t gaussianNoiseNode;

	GaussianNoiseDistanceNodeTest() { gaussianNoiseNode = nullptr; }

	// Traces RAY_COUNT rays from the origin along z through gaussianNoiseNode and returns their distances and hits.
	std::pair<std::vector<float>, std::vector<int32_t>> trace()
	{
		std::vector<rgl_mat3x4f> rays(RAY_COUNT, Mat3x4f::identity().toRGL());
		std::vector<rgl_field_t> fields = {DISTANCE_F32, IS_HIT_I32};
		rgl_node_t useRays = nullptr, raytrace = nullptr, yield = nullptr;
		EXPECT_RGL_SUCCESS(rgl_node_rays_from_mat3x4f(&useRays, rays.data(), rays.size()));
		EXPECT_RGL_SUCCESS(rgl_node_raytrace(&raytrace, nullptr));
		EXPECT_RGL_SUCCESS(rgl_node_raytrace_configure_non_hits(raytrace, NON_HIT_DISTANCE, NON_HIT_DISTANCE));
		EXPECT_RGL_SUCCESS(rgl_node_points_yield(&yield, fields.data(), fields.size()));
		EXPECT_RGL_SUCCESS(rgl_graph_node_add_child(useRays, raytrace));
		EXPECT_RGL_SUCCESS(rgl_graph_node_add_child(raytrace, gaussianNoiseNode));
		EXPECT_RGL_SUCCESS(rgl_graph_node_add_child(gaussianNoiseNode, yield));
		EXPECT_RGL_SUCCESS(rgl_graph_run(raytrace));

		std::vector<float> distances(RAY_COUNT);
		std::vector<int32_t> isHit(RAY_COUNT);
		EXPECT_RGL_SUCCESS(rgl_graph_get_result_data(yield, DISTANCE_F32, distances.data()));
		EXPECT_RGL_SUCCESS(rgl_graph_get_result_data(yield, IS_HIT_I32, isHit.data()));
		return {distances, isHit};
	}

	// A plate 0.2 m thick centred 5 m ahead, its normal tilted from the rays by incidentAngleDeg.
	static void spawnPlate(float incidentAngleDeg)
	{
		spawnCubeOnScene(Mat3x4f::TRS({0, 0, 5}, {incidentAngleDeg, 0, 0}, {50, 50, 0.1f}));
	}
};

TEST_F(GaussianNoiseDistanceNodeTest, invalid_argument_node)
{
	EXPECT_RGL_INVALID_ARGUMENT(rgl_node_gaussian_noise_distance(nullptr, 0.0f, 0.0f, 0.0f, 0.0f, ANGLE_INDEPENDENT),
	                            "node != nullptr");
}

TEST_F(GaussianNoiseDistanceNodeTest, invalid_argument_st_dev_base)
{
	EXPECT_RGL_INVALID_ARGUMENT(
	    rgl_node_gaussian_noise_distance(&gaussianNoiseNode, 0.0f, -1.0f, 0.0f, 0.0f, ANGLE_INDEPENDENT), "st_dev_base >= 0");
}

TEST_F(GaussianNoiseDistanceNodeTest, invalid_argument_st_dev_rise_per_meter)
{
	EXPECT_RGL_INVALID_ARGUMENT(
	    rgl_node_gaussian_noise_distance(&gaussianNoiseNode, 0.0f, 0.0f, -1.0f, 0.0f, ANGLE_INDEPENDENT),
	    "st_dev_rise_per_meter >= 0");
}

TEST_F(GaussianNoiseDistanceNodeTest, invalid_argument_st_dev_rise_per_meter_squared)
{
	EXPECT_RGL_INVALID_ARGUMENT(
	    rgl_node_gaussian_noise_distance(&gaussianNoiseNode, 0.0f, 0.0f, 0.0f, -1.0f, ANGLE_INDEPENDENT),
	    "st_dev_rise_per_meter_squared >= 0");
}

TEST_F(GaussianNoiseDistanceNodeTest, invalid_argument_max_incidence_angle)
{
	EXPECT_RGL_INVALID_ARGUMENT(rgl_node_gaussian_noise_distance(&gaussianNoiseNode, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
	                            "max_incidence_angle > 0");
	EXPECT_RGL_INVALID_ARGUMENT(rgl_node_gaussian_noise_distance(&gaussianNoiseNode, 0.0f, 0.0f, 0.0f, 0.0f, 2.0f),
	                            "max_incidence_angle > 0");
}

TEST_F(GaussianNoiseDistanceNodeTest, valid_arguments)
{
	EXPECT_RGL_SUCCESS(rgl_node_gaussian_noise_distance(&gaussianNoiseNode, 0.1f, 0.1f, 0.01f, 0.001f, ANGLE_INDEPENDENT));

	// If (*gaussianNoiseNode) != nullptr
	EXPECT_RGL_SUCCESS(rgl_node_gaussian_noise_distance(&gaussianNoiseNode, 0.1f, 0.1f, 0.01f, 0.001f, 1.0f));
}

TEST_F(GaussianNoiseDistanceNodeTest, st_dev_grows_with_distance_and_its_square)
{
	constexpr float BASE = 0.01f, RISE = 0.001f, RISE_SQUARED = 0.002f;
	spawnPlate(0.0f);
	ASSERT_RGL_SUCCESS(rgl_node_gaussian_noise_distance(&gaussianNoiseNode, 0.0f, BASE, RISE, RISE_SQUARED, ANGLE_INDEPENDENT));

	auto [distances, isHit] = trace();

	const float distance = 4.9f;
	auto [mean, stDev] = calcMeanAndStdev(distances);
	EXPECT_THAT(mean, testing::FloatNear(distance, EPSILON_NOISE));
	EXPECT_THAT(stDev, testing::FloatNear(BASE + RISE * distance + RISE_SQUARED * distance * distance, EPSILON_NOISE));
}

TEST_F(GaussianNoiseDistanceNodeTest, st_dev_grows_with_incident_angle)
{
	constexpr float BASE = 0.01f, INCIDENT_ANGLE_DEG = 60.0f;
	spawnPlate(INCIDENT_ANGLE_DEG);
	ASSERT_RGL_SUCCESS(rgl_node_gaussian_noise_distance(&gaussianNoiseNode, 0.0f, BASE, 0.0f, 0.0f, 1.4f));

	auto [distances, isHit] = trace();

	EXPECT_THAT(isHit, testing::Each(1));
	auto [mean, stDev] = calcMeanAndStdev(distances);
	EXPECT_THAT(stDev, testing::FloatNear(BASE / cosf(INCIDENT_ANGLE_DEG * static_cast<float>(M_PI) / 180.0f), EPSILON_NOISE));
}

TEST_F(GaussianNoiseDistanceNodeTest, hits_beyond_max_incidence_angle_become_non_hits)
{
	spawnPlate(60.0f);
	ASSERT_RGL_SUCCESS(rgl_node_gaussian_noise_distance(&gaussianNoiseNode, 0.0f, 0.01f, 0.0f, 0.0f, 1.0f));

	auto [distances, isHit] = trace();

	EXPECT_THAT(isHit, testing::Each(0));
	EXPECT_THAT(distances, testing::Each(testing::IsNan()));
}

TEST_F(GaussianNoiseDistanceNodeTest, non_hits_keep_their_values)
{
	ASSERT_RGL_SUCCESS(rgl_node_gaussian_noise_distance(&gaussianNoiseNode, 0.1f, 0.1f, 0.1f, 0.1f, 1.0f));

	auto [distances, isHit] = trace();

	EXPECT_THAT(isHit, testing::Each(0));
	EXPECT_THAT(distances, testing::Each(NON_HIT_DISTANCE));
}
