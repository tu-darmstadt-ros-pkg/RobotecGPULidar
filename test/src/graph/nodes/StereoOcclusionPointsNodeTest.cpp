#include <helpers/commonHelpers.hpp>
#include <helpers/sceneHelpers.hpp>

#include <RGLFields.hpp>
#include <math/Mat3x4f.hpp>

#include <set>

// A camera at the origin looking along z, columns towards +x: a post 1.45 m away (its front face) in front of a wall
// 2.95 m away. With f = 100 px and |b| = 0.21 m the disparities are 14.48 px on the post and 7.12 px on the wall.
class StereoOcclusionPointsNodeTest : public RGLTest
{
protected:
	static constexpr int WIDTH = 101;
	static constexpr int HEIGHT = 3;
	static constexpr float FOCAL_LENGTH = 100.0f;
	static constexpr float BASELINE = 0.21f;
	static constexpr rgl_vec3f OPTICAL_AXIS = {0.0f, 0.0f, 1.0f};

	rgl_node_t occlusionNode;

	StereoOcclusionPointsNodeTest() { occlusionNode = nullptr; }

	// The columns, in every row, that the node turns into non-hits; and whether their distances became NaN.
	std::set<int> unmeasuredColumns(float baseline, int matchingBand)
	{
		spawnCubeOnScene(Mat3x4f::TRS({0, 0, 1.5f}, {0, 0, 0}, {0.1f, 50, 0.05f}));
		spawnCubeOnScene(Mat3x4f::TRS({0, 0, 3.0f}, {0, 0, 0}, {50, 50, 0.05f}));

		std::vector<rgl_mat3x4f> rays;
		for (int row = 0; row < HEIGHT; ++row) {
			for (int column = 0; column < WIDTH; ++column) {
				const float angle = atanf((static_cast<float>(column) - (WIDTH - 1) / 2.0f) / FOCAL_LENGTH);
				rays.push_back(Mat3x4f::rotationRad(0, angle, 0).toRGL());
			}
		}
		std::vector<rgl_field_t> fields = {IS_HIT_I32, DISTANCE_F32};
		rgl_node_t useRays = nullptr, raytrace = nullptr, yield = nullptr;
		EXPECT_RGL_SUCCESS(
		    rgl_node_points_stereo_occlusion(&occlusionNode, WIDTH, FOCAL_LENGTH, baseline, &OPTICAL_AXIS, matchingBand));
		EXPECT_RGL_SUCCESS(rgl_node_rays_from_mat3x4f(&useRays, rays.data(), rays.size()));
		EXPECT_RGL_SUCCESS(rgl_node_raytrace(&raytrace, nullptr));
		EXPECT_RGL_SUCCESS(rgl_node_points_yield(&yield, fields.data(), fields.size()));
		EXPECT_RGL_SUCCESS(rgl_graph_node_add_child(useRays, raytrace));
		EXPECT_RGL_SUCCESS(rgl_graph_node_add_child(raytrace, occlusionNode));
		EXPECT_RGL_SUCCESS(rgl_graph_node_add_child(occlusionNode, yield));
		EXPECT_RGL_SUCCESS(rgl_graph_run(raytrace));

		std::vector<int32_t> isHit(rays.size());
		std::vector<float> distances(rays.size());
		EXPECT_RGL_SUCCESS(rgl_graph_get_result_data(yield, IS_HIT_I32, isHit.data()));
		EXPECT_RGL_SUCCESS(rgl_graph_get_result_data(yield, DISTANCE_F32, distances.data()));

		std::set<int> columns;
		for (int column = 0; column < WIDTH; ++column) {
			if (!isHit[column]) {
				columns.insert(column);
				EXPECT_TRUE(std::isnan(distances[column]));
			}
			for (int row = 1; row < HEIGHT; ++row) {
				EXPECT_EQ(isHit[row * WIDTH + column], isHit[column]);
			}
		}
		return columns;
	}

	static std::set<int> range(int first, int last)
	{
		std::set<int> out;
		for (int i = first; i <= last; ++i) {
			out.insert(i);
		}
		return out;
	}

	static std::set<int> unite(const std::set<int>& a, const std::set<int>& b)
	{
		std::set<int> out = a;
		out.insert(b.begin(), b.end());
		return out;
	}
};

TEST_F(StereoOcclusionPointsNodeTest, invalid_arguments)
{
	EXPECT_RGL_INVALID_ARGUMENT(rgl_node_points_stereo_occlusion(nullptr, WIDTH, FOCAL_LENGTH, BASELINE, &OPTICAL_AXIS, 0),
	                            "node != nullptr");
	EXPECT_RGL_INVALID_ARGUMENT(rgl_node_points_stereo_occlusion(&occlusionNode, 0, FOCAL_LENGTH, BASELINE, &OPTICAL_AXIS, 0),
	                            "width > 0");
	EXPECT_RGL_INVALID_ARGUMENT(rgl_node_points_stereo_occlusion(&occlusionNode, WIDTH, 0.0f, BASELINE, &OPTICAL_AXIS, 0),
	                            "focal_length > 0");
	EXPECT_RGL_INVALID_ARGUMENT(rgl_node_points_stereo_occlusion(&occlusionNode, WIDTH, FOCAL_LENGTH, BASELINE, nullptr, 0),
	                            "optical_axis != nullptr");
	EXPECT_RGL_INVALID_ARGUMENT(
	    rgl_node_points_stereo_occlusion(&occlusionNode, WIDTH, FOCAL_LENGTH, BASELINE, &OPTICAL_AXIS, -1),
	    "matching_band >= 0");
}

TEST_F(StereoOcclusionPointsNodeTest, valid_arguments)
{
	EXPECT_RGL_SUCCESS(rgl_node_points_stereo_occlusion(&occlusionNode, WIDTH, FOCAL_LENGTH, BASELINE, &OPTICAL_AXIS, 1));

	// If (*occlusionNode) != nullptr
	EXPECT_RGL_SUCCESS(rgl_node_points_stereo_occlusion(&occlusionNode, WIDTH, FOCAL_LENGTH, -BASELINE, &OPTICAL_AXIS, 2));
}

// The post covers columns 44 to 56. The second camera to the right does not see the wall on the post's left, where
// u - 7.12 > 44 - 14.48 (columns 37 to 43), nor the image's left edge, where u - 7.12 < -0.5 (columns 0 to 6).
TEST_F(StereoOcclusionPointsNodeTest, removes_what_the_second_camera_does_not_see)
{
	EXPECT_EQ(unmeasuredColumns(BASELINE, 0), unite(range(0, 6), range(37, 43)));
}

TEST_F(StereoOcclusionPointsNodeTest, second_camera_on_the_left_mirrors_the_shadow)
{
	EXPECT_EQ(unmeasuredColumns(-BASELINE, 0), unite(range(57, 63), range(94, 100)));
}

TEST_F(StereoOcclusionPointsNodeTest, matching_band_widens_the_shadow)
{
	EXPECT_EQ(unmeasuredColumns(BASELINE, 1), unite(range(0, 7), range(36, 44)));
}

TEST_F(StereoOcclusionPointsNodeTest, rows_must_fill_the_width)
{
	std::vector<rgl_mat3x4f> rays(WIDTH * HEIGHT, Mat3x4f::identity().toRGL());
	std::vector<rgl_field_t> fields = {IS_HIT_I32};
	rgl_node_t useRays = nullptr, raytrace = nullptr, yield = nullptr;
	ASSERT_RGL_SUCCESS(rgl_node_points_stereo_occlusion(&occlusionNode, WIDTH - 1, FOCAL_LENGTH, BASELINE, &OPTICAL_AXIS, 0));
	ASSERT_RGL_SUCCESS(rgl_node_rays_from_mat3x4f(&useRays, rays.data(), rays.size()));
	ASSERT_RGL_SUCCESS(rgl_node_raytrace(&raytrace, nullptr));
	ASSERT_RGL_SUCCESS(rgl_node_points_yield(&yield, fields.data(), fields.size()));
	ASSERT_RGL_SUCCESS(rgl_graph_node_add_child(useRays, raytrace));
	ASSERT_RGL_SUCCESS(rgl_graph_node_add_child(raytrace, occlusionNode));
	ASSERT_RGL_SUCCESS(rgl_graph_node_add_child(occlusionNode, yield));
	ASSERT_RGL_SUCCESS(rgl_graph_run(raytrace));

	int32_t count = 0, size = 0;
	EXPECT_RGL_INVALID_PIPELINE(rgl_graph_get_result_size(yield, IS_HIT_I32, &count, &size), "do not fill rows");
}
