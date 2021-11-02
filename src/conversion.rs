use na::{Point2, Vector2, Isometry2, Point};

pub fn convert_to_vec_of_vectors(vector: gdnative::core_types::Vector2Array) -> Vec<Vector2<f32>> {
    let mut rust_vec = Vec::new();
    for value in vector.read().iter() {
        rust_vec.push(Vector2::new(value.x, value.y));
    }
    return rust_vec;
}

pub fn convert_to_points(godot_vector: gdnative::core_types::Vector2Array, sim_scaling_factor: f32) -> Vec<Point2<f32>> {
    let mut points = Vec::new();
    for point in godot_vector.read().iter() {
        points.push(Point2::new(point.x * sim_scaling_factor, point.y * sim_scaling_factor));
    }
    return points;
}
