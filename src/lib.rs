extern crate nalgebra as na;

use gdnative::prelude::*;

use na::{Point2, Vector2, Isometry2, Point};
use ncollide2d::shape::{ShapeHandle, Polyline, ConvexPolygon};
use nphysics2d::force_generator::DefaultForceGeneratorSet;
use nphysics2d::joint::DefaultJointConstraintSet;
use nphysics2d::object::{BodyPartHandle, ColliderDesc, DefaultBodySet, DefaultColliderSet, RigidBodyDesc, BodyStatus, DefaultColliderHandle, DefaultBodyHandle};
use nphysics2d::world::{DefaultGeometricalWorld, DefaultMechanicalWorld};
use salva2d::coupling::{ColliderCouplingSet, CouplingMethod};
use salva2d::object::{Fluid, Boundary, BoundaryHandle, FluidHandle, ContiguousArenaIndex, FluidSet};
use salva2d::solver::{ArtificialViscosity, IISPHSolver};
use salva2d::LiquidWorld;
use ncollide2d::bounding_volume::HasBoundingVolume;
use ncollide2d::query::PointQuery;
use nphysics2d::algebra::{Force2, ForceType, Velocity2};
use contour::ContourBuilder;
use geojson::Value;
use std::cmp;
use nphysics2d::material::{MaterialHandle, BasicMaterial};
use std::convert::TryInto;


mod conversion;


const SIM_SCALING_FACTOR: f32 = 0.02;

#[derive(NativeClass)]
#[inherit(Node)]
struct Physics {
    bodies: DefaultBodySet<f32>,
    colliders: DefaultColliderSet<f32>,
    objects: Vec<gdnative::api::Polygon2D>,
    mechanical_world: DefaultMechanicalWorld<f32>,
    geometrical_world: DefaultGeometricalWorld<f32>,
    liquid_world: LiquidWorld<f32>,
    coupling_set: ColliderCouplingSet<f32, DefaultColliderHandle>,
    joint_constraints: DefaultJointConstraintSet<f32>,
    force_generators: DefaultForceGeneratorSet<f32>,
    particle_rad: f32
}

#[methods]
impl Physics {
    fn new(_owner: &Node) -> Self {
        let rad = 5.0 * SIM_SCALING_FACTOR;
        Physics {
            bodies: DefaultBodySet::new(),
            colliders: DefaultColliderSet::new(),
            objects: Vec::new(),
            mechanical_world: DefaultMechanicalWorld::new(Vector2::new(0.0, 9.81)),
            geometrical_world: DefaultGeometricalWorld::new(),
            liquid_world: LiquidWorld::new(IISPHSolver::<f32>::new(), rad, 2.0),
            coupling_set: ColliderCouplingSet::new(),
            joint_constraints: DefaultJointConstraintSet::new(),
            force_generators: DefaultForceGeneratorSet::new(),
            particle_rad: rad,
        }
    }

    #[export]
    fn add_rigid_body(&mut self, _owner: &Node, position: gdnative::core_types::Vector2, polygon: gdnative::core_types::Vector2Array, mass: f32, density: f32, restitution: f32, friction: f32, body_status: i32) -> Vec<usize> {
        let mut status = BodyStatus::Dynamic;
        if body_status == 1 {
            status = BodyStatus::Static;
        } else if body_status == 2 {
            status = BodyStatus::Kinematic;
        }
        let rb = RigidBodyDesc::new().status(status).
            mass(mass).
            angular_inertia(0.0).
            rotation(0.0).
            local_center_of_mass(Point2::new(0.0, 0.0)).
            translation(Vector2::new(position.x * SIM_SCALING_FACTOR, position.y * SIM_SCALING_FACTOR)).build();

        let rb_handle = self.bodies.insert(rb);

        // Build the collider.
        let geom = ShapeHandle::new(self.convert_polygon2(polygon));
        let geom_sample =
            salva2d::sampling::shape_surface_ray_sample(&*geom, self.particle_rad).unwrap();
        let co = ColliderDesc::new(geom)
            //.margin(0.3)
            .density(density)
            .material(MaterialHandle::new(BasicMaterial::new(restitution, friction)))
            .build(BodyPartHandle(rb_handle, 0));
        let co_handle = self.colliders.insert(co);
        let bo_handle = self.liquid_world.add_boundary(Boundary::new(Vec::new()));
        self.coupling_set.register_coupling(
            bo_handle,
            co_handle,
            CouplingMethod::DynamicContactSampling,
        );
        let (index, generation) = rb_handle.into_raw_parts();
        let (collider_index, collider_generation) = co_handle.into_raw_parts();
        let mut indices = Vec::new();
        indices.push(index);
        indices.push(collider_index);
        return indices;
    }

    #[export]
    fn deactivate_rigid_body(&mut self, _owner: &Node, index: usize) {
        let body = self.bodies.get_mut(DefaultBodyHandle::from_raw_parts(index, 0)).unwrap();
        body.deactivate();
        body.set_status(BodyStatus::Disabled);
    }

    #[export]
    fn activate_rigid_body(&mut self, _owner: &Node, index: usize) {
        let body = self.bodies.get_mut(DefaultBodyHandle::from_raw_parts(index, 0)).unwrap();
        body.activate();
        body.set_status(BodyStatus::Dynamic);
    }

    #[export]
    fn deactivate_liquid_coupling(&mut self, _owner: &Node, collider_index: usize) {
        let boundary_handle = self.coupling_set.unregister_coupling(DefaultColliderHandle::from_raw_parts(collider_index, 0)).unwrap();
        self.liquid_world.remove_boundary(boundary_handle);
    }

    #[export]
    fn activate_liquid_coupling(&mut self, _owner: &Node, collider_index: usize) {
        let bo_handle = self.liquid_world.add_boundary(Boundary::new(Vec::new()));
        self.coupling_set.register_coupling(
            bo_handle,
            DefaultBodyHandle::from_raw_parts(collider_index, 0),
            CouplingMethod::DynamicContactSampling,
        );
    }

    #[export]
    fn add_sensor_to_body(&mut self, _owner: &Node, body_index: usize, position: gdnative::core_types::Vector2, polygon: gdnative::core_types::Vector2Array) -> usize {
        let body_handle = DefaultBodyHandle::from_raw_parts(body_index, 0);
        let sensor_geom = ShapeHandle::new(self.convert_polygon2(polygon));
        let sensor_collider = ColliderDesc::new(sensor_geom)
            .sensor(true)
            .build(BodyPartHandle(body_handle, 0));
        let collider_handle = self.colliders.insert(sensor_collider);
        let (collider_index, generation) = collider_handle.into_raw_parts();
        return collider_index;
    }

    #[export]
    fn add_sensor(&mut self, _owner: &Node, position: gdnative::core_types::Vector2, polygon: gdnative::core_types::Vector2Array) -> usize {
        let rb = RigidBodyDesc::new().status(BodyStatus::Kinematic).
            mass(1.0).
            angular_inertia(0.0).
            rotation(0.0).
            local_center_of_mass(Point2::new(0.0, 0.0)).
            translation(Vector2::new(position.x * SIM_SCALING_FACTOR, position.y * SIM_SCALING_FACTOR)).build();

        let body_handle = self.bodies.insert(rb);
        let sensor_geom = ShapeHandle::new(self.convert_polygon2(polygon));
        let sensor_collider = ColliderDesc::new(sensor_geom)
            .sensor(true)
            .build(BodyPartHandle(body_handle, 0));
        let collider_handle = self.colliders.insert(sensor_collider);
        let (collider_index, generation) = collider_handle.into_raw_parts();
        return collider_index;
    }

    #[export]
    fn get_contacting_colliders(&mut self, _owner: &Node, collider_index: usize) -> Vec<usize> {
        let mut collider_indices = Vec::new();
        for stuff in self.geometrical_world.colliders_in_proximity_of(&self.colliders, DefaultColliderHandle::from_raw_parts(collider_index, 0)).unwrap() {
            let (handle, collider) = stuff;
            if self.bodies.get_mut(collider.body()).unwrap().status() != BodyStatus::Disabled && !collider.is_sensor() {
                let (index, generation) = handle.into_raw_parts();
                collider_indices.push(index);
            }
        }
        return collider_indices;
    }

    // fn get_contacting_particles(&mut self, owner: &Node) {
    //for fluid in self.liquid_world.fluids().iter() {
    //for particle in fluid {
    //  particle.f
    //}
    //  }
    //}

    #[export]
    fn apply_force(&mut self, _owner: &Node, force: gdnative::core_types::Vector2, angular_force: f32, index: usize) {
        let body = self.bodies.get_mut(DefaultBodyHandle::from_raw_parts(index, 0)).unwrap();
        body.apply_force(0, &Force2::new(Vector2::new(force.x, force.y), angular_force), ForceType::Force, true);
    }

    #[export]
    fn set_velocity(&mut self, _owner: &Node, velocity: gdnative::core_types::Vector2, angular_force: f32, index: usize) {
        let body = self.bodies.rigid_body_mut(DefaultBodyHandle::from_raw_parts(index, 0)).unwrap();
        body.set_velocity(Velocity2::new(Vector2::new(velocity.x, velocity.y), angular_force));
    }

    #[export]
    fn get_velocity(&mut self, _owner: &Node, index: usize) -> gdnative::core_types::Vector2 {
        let body = self.bodies.rigid_body_mut(DefaultBodyHandle::from_raw_parts(index, 0)).unwrap();
        return gdnative::core_types::Vector2::new(body.velocity().linear.x, body.velocity().linear.y);
    }

    #[export]
    fn get_angular_velocity(&mut self, _owner: &Node, index: usize) -> f32 {
        let body = self.bodies.rigid_body_mut(DefaultBodyHandle::from_raw_parts(index, 0)).unwrap();
        return body.velocity().angular;
    }

    #[export]
    fn set_angular_velocity(&mut self, _owner: &Node, index: usize, angular_velocity: f32) {
        let body = self.bodies.rigid_body_mut(DefaultBodyHandle::from_raw_parts(index, 0)).unwrap();
        return body.set_angular_velocity(angular_velocity);
    }

    #[export]
    fn set_position(&mut self, _owner: &Node, position: gdnative::core_types::Vector2, angle: f32, index: usize) {
        let body = self.bodies.rigid_body_mut(DefaultBodyHandle::from_raw_parts(index, 0)).unwrap();
        body.set_position(Isometry2::new(Vector2::new(position.x * SIM_SCALING_FACTOR, position.y * SIM_SCALING_FACTOR), angle));
    }

    #[export]
    fn get_position(&mut self, _owner: &Node, index: usize) -> gdnative::core_types::Vector2 {
        let body = self.bodies.rigid_body_mut(DefaultBodyHandle::from_raw_parts(index, 0)).unwrap();
        return gdnative::core_types::Vector2::new(body.position().translation.x / SIM_SCALING_FACTOR, body.position().translation.y / SIM_SCALING_FACTOR);
    }

    #[export]
    fn get_rotation(&mut self, _owner: &Node, index: usize) -> f32 {
        let body = self.bodies.rigid_body_mut(DefaultBodyHandle::from_raw_parts(index, 0)).unwrap();
        return body.position().rotation.angle();
    }

    #[export]
    fn set_mass(&mut self, _owner: &Node, mass: f32, index: usize) {
        let body = self.bodies.rigid_body_mut(DefaultBodyHandle::from_raw_parts(index, 0)).unwrap();
        body.set_mass(mass);
    }

    #[export]
    fn set_angular_damping(&mut self, _owner: &Node, angular_damping: f32, index: usize) {
        let body = self.bodies.rigid_body_mut(DefaultBodyHandle::from_raw_parts(index, 0)).unwrap();
        body.set_angular_damping(angular_damping);
    }

    #[export]
    fn set_angular_inertia(&mut self, _owner: &Node, angular_inertia: f32, index: usize) {
        let body = self.bodies.rigid_body_mut(DefaultBodyHandle::from_raw_parts(index, 0)).unwrap();
        body.set_angular_inertia(angular_inertia);
    }

    #[export]
    fn add_liquid(&mut self, _owner: &Node, droplets: gdnative::core_types::Vector2Array, velocities: gdnative::core_types::Vector2Array) -> gdnative::core_types::Vector2 {
        let mut points = conversion::convert_to_points(droplets, SIM_SCALING_FACTOR);

        let viscosity = ArtificialViscosity::new(0.5, 0.0);
        let mut fluid = Fluid::new(points, self.particle_rad, 1.0);
        fluid.velocities = conversion::convert_to_vec_of_vectors(velocities);
        fluid.nonpressure_forces.push(Box::new(viscosity.clone()));
        let fluid_handle: FluidHandle = self.liquid_world.add_fluid(fluid);
        let idx: ContiguousArenaIndex = fluid_handle.into();
        let (fluid_index, generation) = idx.into_raw_parts();
        return gdnative::core_types::Vector2::new(fluid_index as f32, generation as f32);
    }

    #[export]
    fn remove_liquid(&mut self, _owner: &Node, liquid_index: gdnative::core_types::Vector2) {
        let fluid_handle = self.get_liquid_handle(liquid_index);
        self.liquid_world.remove_fluid(fluid_handle);
    }

    #[export]
    fn get_liquid(&mut self, _owner: &Node) -> Vector2Array {
        let mut droplets = Vector2Array::new();
        for (i, fluid) in self.liquid_world.fluids().iter() {
            for droplet in &fluid.positions {
                droplets.push(gdnative::core_types::Vector2::new(droplet.x / SIM_SCALING_FACTOR, droplet.y / SIM_SCALING_FACTOR));
            }
        }
        return droplets;
    }

    #[export]
    fn move_particles(&mut self, _owner: &Node, fluid_index: gdnative::core_types::Vector2, particles_indices: Vec<usize>, new_positions: Vector2Array, new_velocities: Vector2Array) {
        let mut fluid = self.get_mutable_liquid_by_index(fluid_index);

        let mut positions = vec![Point2::new(0., 0.); new_positions.len() as usize];
        let mut velocities = vec![Vector2::new(0., 0.); new_positions.len() as usize];
        for particle_index in particles_indices {
            let position = positions.get(particle_index).unwrap();
            let velocity = velocities.get(particle_index).unwrap();
            positions[particle_index] = Point2::new(position.x, position.y);
            velocities[particle_index] = Vector2::new(velocity.x, velocity.y);
            fluid.delete_particle_at_next_timestep(particle_index as usize);
        }
        fluid.add_particles(&positions[..], Some(&velocities[..]));
    }

    #[export]
    fn remove_particles(&mut self, _owner: &Node, fluid_index: gdnative::core_types::Vector2, collider_index: usize) {
        let particle_indices = self.get_contacting_liquid_indices(_owner, collider_index);
        let mut fluid = self.get_mutable_liquid_by_index(fluid_index);

        for particle_index in particle_indices {
            fluid.delete_particle_at_next_timestep(particle_index as usize);
        }
    }

    #[export]
    fn add_particles(&mut self, _owner: &Node, fluid_index: gdnative::core_types::Vector2, new_positions: Vector2Array, new_velocities: Vector2Array) {
        let mut fluid = self.get_mutable_liquid_by_index(fluid_index);
        let positions = conversion::convert_to_points(new_positions, SIM_SCALING_FACTOR);
        let velocities = conversion::convert_to_vec_of_vectors(new_velocities);
        fluid.add_particles(&positions[..], Some(&velocities[..]));
    }

    #[export]
    fn get_liquid_raster(&mut self, _owner: &Node, x_min: f32, x_max: f32, y_min: f32, y_max: f32, resolution: f32, meta_ball_influence: i64) -> Ref<gdnative::api::Image, Unique> {
        let rgba8 = 5;
        let width = ((x_max - x_min) / resolution + 1.0).ceil() as i64;
        let height = ((y_max - y_min) / resolution + 1.0).ceil() as i64;
        let mut data = vec![vec![0.0f32; height as usize]; width as usize];

        let mut image = gdnative::api::Image::new();
        image.create(width, height, false, rgba8);
        image.lock();
        image.fill(gdnative::core_types::Color::rgba(0., 0.0, 0.0, 0.0));
        image.unlock();
        let scaled_x_min = x_min * SIM_SCALING_FACTOR;
        let scaled_x_max = x_max * SIM_SCALING_FACTOR;
        let scaled_y_min = y_min * SIM_SCALING_FACTOR;
        let scaled_y_max = y_max * SIM_SCALING_FACTOR;

        for (i, fluid) in self.liquid_world.fluids().iter() {
            for droplet in &fluid.positions {
                if droplet.x >= scaled_x_min && droplet.x <= scaled_x_max && droplet.y >= scaled_y_min && droplet.y <= scaled_y_max {
                    let true_x = (droplet.x / SIM_SCALING_FACTOR - x_min) / resolution;
                    let x = true_x.round() as i64;
                    let x_start = cmp::max(0, x - meta_ball_influence);
                    let x_end = cmp::min(width, x + meta_ball_influence);

                    let true_y = ((droplet.y / SIM_SCALING_FACTOR - y_min) / resolution);
                    let y = true_y.round() as i64;
                    let y_start = cmp::max(0, y - meta_ball_influence);
                    let y_end = cmp::min(height, y + meta_ball_influence);

                    for x_i in x_start..x_end {
                        for y_i in y_start..y_end {
                            data[x_i as usize][y_i as usize] += 1. / ((true_x - x_i as f32).powf(2.0) + (true_y - y_i as f32).powf(2.0));
                        }
                    }
                }
            }
        }

        image.lock();
        for x_i in 0..width {
            for y_i in 0..height {
                image.set_pixel(
                    x_i,
                    y_i,
                    gdnative::core_types::Color::rgba(0., 0.1, 0.8, data[x_i as usize][y_i as usize]),
                );
            }
        }

        image.unlock();
        return image;
    }

    #[export]
    fn get_liquid_as_polygons(&mut self, _owner: &Node) -> Vec<Vector2Array> {
        //let droplets = self.get_liquid(owner);

        let res = 0.2;

        let mut min_x = f32::MAX;
        let mut max_x = f32::MIN;
        let mut min_y = f32::MAX;
        let mut max_y = f32::MIN;


        for (i, fluid) in self.liquid_world.fluids().iter() {
            for droplet in &fluid.positions {
                if droplet.x <= min_x {
                    min_x = droplet.x;
                } else if droplet.x >= max_x {
                    max_x = droplet.x;
                }

                if droplet.y <= min_y {
                    min_y = droplet.y;
                } else if droplet.y >= max_y {
                    max_y = droplet.y;
                }
            }
        }

        let x_length = ((max_x - min_x) / res).ceil();
        let y_length = ((max_y - min_y) / res).ceil();

        let c = ContourBuilder::new(
            x_length as u32,
            y_length as u32,
            false,
        );
        let number_of_elements = x_length * y_length;
        let mut field = vec![0.0; number_of_elements as usize];

        for (i, fluid) in self.liquid_world.fluids().iter() {
            for droplet in &fluid.positions {
                field[(((droplet.x - min_x) / res).floor() + ((droplet.y - min_y) / res).floor() * x_length) as usize] = 1.0;
            }
        }

        let polygons = c.contours(&field, &[0.5]);
        let mut result = Vec::new();
        for feature in polygons.unwrap() {
            let multi_poly = feature.geometry.unwrap().value;
            match multi_poly {
                Value::MultiPolygon(ref collection) =>
                    for fake_poly in collection {
                        for poly in fake_poly {
                            let mut poly_result = Vector2Array::new();
                            for point in poly {
                                poly_result.push(
                                    gdnative::core_types::Vector2::new(
                                        (point[0] as f32 * res + min_x) / SIM_SCALING_FACTOR,
                                        (point[1] as f32 * res + min_y) / SIM_SCALING_FACTOR,
                                    )
                                );
                            }
                            result.push(poly_result);
                        }
                    }
                _ => println!("accidents happen"),
            }
        }
        return result;
    }

    #[export]
    fn get_liquid_velocities(&mut self, _owner: &Node, liquid_index: gdnative::core_types::Vector2) -> Vector2Array {
        let mut droplets = Vector2Array::new();
        for droplet in &self.get_liquid_by_index(liquid_index).velocities {
            droplets.push(gdnative::core_types::Vector2::new(droplet.x, droplet.y));
        }
        return droplets;
    }

    #[export]
    fn get_all_liquid_velocities(&mut self, _owner: &Node) -> Vector2Array {
        let mut velocities = Vector2Array::new();
        for (i, fluid) in self.liquid_world.fluids().iter() {
            for velocity in &fluid.velocities {
                velocities.push(gdnative::core_types::Vector2::new(velocity.x / SIM_SCALING_FACTOR, velocity.y / SIM_SCALING_FACTOR));
            }
        }
        return velocities;
    }

    fn get_liquid_by_index(&self, liquid_index: gdnative::core_types::Vector2) -> &Fluid<f32> {
        let fluid_handle = self.get_liquid_handle(liquid_index);
        return self.liquid_world.fluids().get(fluid_handle).unwrap();
    }

    fn get_mutable_liquid_by_index(&mut self, liquid_index: gdnative::core_types::Vector2) -> &mut Fluid<f32> {
        let fluid_handle = self.get_liquid_handle(liquid_index);
        return self.liquid_world.fluids_mut().get_mut(fluid_handle).unwrap();
    }

    fn get_liquid_handle(&self, liquid_index: gdnative::core_types::Vector2) -> FluidHandle {
        let index = ContiguousArenaIndex::from_raw_parts(liquid_index.x as usize, liquid_index.y as u64);
        return FluidHandle::from(index);
    }

    #[export]
    fn get_contacting_liquids(&mut self, _owner: &Node, collider_index: usize) -> Vector2Array {
        let mut droplets = Vector2Array::new();
        let mut collider = self.colliders.get(DefaultColliderHandle::from_raw_parts(collider_index, 0)).unwrap();
        let mut shape = collider.shape();
        let isometry = collider.position();

        for (i, fluid) in self.liquid_world.fluids().iter() {
            for droplet in &fluid.positions {
                if shape.contains_point(isometry, droplet) {
                    droplets.push(gdnative::core_types::Vector2::new(droplet.x / SIM_SCALING_FACTOR, droplet.y / SIM_SCALING_FACTOR));
                }
            }
        }
        return droplets;
    }

    #[export]
    fn get_contacting_liquid_indices(&mut self, _owner: &Node, collider_index: usize) -> Vec<usize> {
        let mut droplet_indices = Vec::new();
        let mut collider = self.colliders.get(DefaultColliderHandle::from_raw_parts(collider_index, 0)).unwrap();
        let mut shape = collider.shape();
        let isometry = collider.position();

        for (i, fluid) in self.liquid_world.fluids().iter() {
            let mut droplet_index = 0;
            for droplet in &fluid.positions {
                if shape.contains_point(isometry, droplet) {
                    droplet_indices.push(droplet_index);
                }
                droplet_index += 1;
            }
        }
        return droplet_indices;
    }

    #[export]
    fn get_polygons(&mut self, _owner: &Node) -> Vector3Array {
        let mut polygons = Vector3Array::new();
        for (i, polygon) in self.colliders.iter() {
            let position: &Isometry2<f32> = polygon.position();
            polygons.push(Vector3::new(position.translation.x / SIM_SCALING_FACTOR, position.translation.y / SIM_SCALING_FACTOR, position.rotation.angle()))
        }
        return polygons;
    }

    fn convert_polygon(&self, polygon: gdnative::core_types::Vector2Array) -> Polyline<f32> {
        let mut points = conversion::convert_to_points(polygon, SIM_SCALING_FACTOR);
        return Polyline::new(points, None);
    }

    fn convert_polygon2(&self, polygon: gdnative::core_types::Vector2Array) -> ConvexPolygon<f32> {
        let mut points = conversion::convert_to_points(polygon, SIM_SCALING_FACTOR);
        return ConvexPolygon::try_from_points(&points).unwrap();
    }

    #[export]
    fn _process(&mut self, _owner: &Node, delta: f32) {
        self.mechanical_world.step(
            &mut self.geometrical_world,
            &mut self.bodies,
            &mut self.colliders,
            &mut self.joint_constraints,
            &mut self.force_generators,
        );


        let dt = self.mechanical_world.timestep();
        let gravity = &self.mechanical_world.gravity;
        self.liquid_world.step_with_coupling(
            dt,
            gravity,
            &mut self.coupling_set
                .as_manager_mut(&self.colliders, &mut self.bodies),
        );
    }
}


fn init(handle: InitHandle) {
    handle.add_class::<Physics>();
}

godot_gdnative_init!();
godot_nativescript_init!(init);
godot_gdnative_terminate!();
