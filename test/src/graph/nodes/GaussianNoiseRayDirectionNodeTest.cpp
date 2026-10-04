#include <helpers/commonHelpers.hpp>
#include <helpers/mathHelpers.hpp>
#include <helpers/sceneHelpers.hpp>

#include <RGLFields.hpp>
#include <math/Mat3x4f.hpp>

class GaussianNoiseRayDirectionNodeTest : public RGLTest
{
protected:
	rgl_node_t noiseNode;

	GaussianNoiseRayDirectionNodeTest() { noiseNode = nullptr; }
};

TEST_F(GaussianNoiseRayDirectionNodeTest, invalid_arguments)
{
	EXPECT_RGL_INVALID_ARGUMENT(rgl_node_gaussian_noise_ray_direction(nullptr, 0.0f), "node != nullptr");
	EXPECT_RGL_INVALID_ARGUMENT(rgl_node_gaussian_noise_ray_direction(&noiseNode, -1.0f), "st_dev >= 0");
}

TEST_F(GaussianNoiseRayDirectionNodeTest, valid_arguments)
{
	EXPECT_RGL_SUCCESS(rgl_node_gaussian_noise_ray_direction(&noiseNode, 0.01f));

	// If (*noiseNode) != nullptr
	EXPECT_RGL_SUCCESS(rgl_node_gaussian_noise_ray_direction(&noiseNode, 0.02f));
}

TEST_F(GaussianNoiseRayDirectionNodeTest, tilts_each_ray_across_its_direction)
{
	constexpr int RAY_COUNT = 100000;
	constexpr float ST_DEV = 0.01f;
	constexpr float EPSILON_NOISE = 0.0005f;

	// Rays along x, onto a wall 5 m away: the tilts show up in y and z.
	spawnCubeOnScene(Mat3x4f::TRS({5, 0, 0}, {0, 0, 0}, {0.1f, 50, 50}));
	std::vector<rgl_mat3x4f> rays(RAY_COUNT, Mat3x4f::rotationDeg(0, 90, 0).toRGL());
	std::vector<rgl_field_t> fields = {XYZ_VEC3_F32};
	rgl_node_t useRays = nullptr, raytrace = nullptr, yield = nullptr;
	ASSERT_RGL_SUCCESS(rgl_node_gaussian_noise_ray_direction(&noiseNode, ST_DEV));
	ASSERT_RGL_SUCCESS(rgl_node_rays_from_mat3x4f(&useRays, rays.data(), rays.size()));
	ASSERT_RGL_SUCCESS(rgl_node_raytrace(&raytrace, nullptr));
	ASSERT_RGL_SUCCESS(rgl_node_points_yield(&yield, fields.data(), fields.size()));
	ASSERT_RGL_SUCCESS(rgl_graph_node_add_child(useRays, noiseNode));
	ASSERT_RGL_SUCCESS(rgl_graph_node_add_child(noiseNode, raytrace));
	ASSERT_RGL_SUCCESS(rgl_graph_node_add_child(raytrace, yield));
	ASSERT_RGL_SUCCESS(rgl_graph_run(raytrace));

	std::vector<Vec3f> points(RAY_COUNT);
	ASSERT_RGL_SUCCESS(rgl_graph_get_result_data(yield, XYZ_VEC3_F32, points.data()));
	std::vector<float> horizontal(RAY_COUNT), vertical(RAY_COUNT);
	for (int i = 0; i < RAY_COUNT; ++i) {
		horizontal[i] = atan2f(points[i].y(), points[i].x());
		vertical[i] = atan2f(points[i].z(), points[i].x());
	}

	for (const auto& angles : {horizontal, vertical}) {
		auto [mean, stDev] = calcMeanAndStdev(angles);
		EXPECT_THAT(mean, testing::FloatNear(0.0f, EPSILON_NOISE));
		EXPECT_THAT(stDev, testing::FloatNear(ST_DEV, EPSILON_NOISE));
	}
}
