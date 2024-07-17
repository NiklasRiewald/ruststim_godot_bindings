use na::{Point2, Vector2};

pub fn convert_to_vec_of_vectors(vector: godot::builtin::Array<godot::builtin::Vector2>) -> Vec<Vector2<f32>> {
    let mut rust_vec = Vec::new();
    for value in vector.iter_shared() {
        rust_vec.push(Vector2::new(value.x, value.y));
    }
    return rust_vec;
}

pub fn convert_to_points(godot_vector: godot::builtin::Array<godot::builtin::Vector2>, sim_scaling_factor: f32) -> Vec<Point2<f32>> {
    let mut points = Vec::new();
    for point in godot_vector.iter_shared() {
        points.push(Point2::new(point.x * sim_scaling_factor, point.y * sim_scaling_factor));
    }
    return points;
}
