extern crate nalgebra as na;

use gdnative::prelude::*;

use na::{Point2, Vector2, Isometry2};
use salva2d::object::{Fluid, Boundary, FluidHandle, ContiguousArenaIndex};
use salva2d::solver::ArtificialViscosity;
use std::cmp;
use salva2d::integrations::rapier::{FluidsPipeline, ColliderSampling, ColliderCouplingSet};
use salva2d::rapier::dynamics::{RigidBodySet, IslandManager, IntegrationParameters, JointSet, CCDSolver, RigidBodyType, CoefficientCombineRule, RigidBodyHandle, RigidBodyBuilder, RigidBody};
use salva2d::rapier::geometry::{ColliderSet, BroadPhase, NarrowPhase, ColliderBuilder, ColliderHandle, Collider, InteractionGroups};
use salva2d::rapier::pipeline::{PhysicsPipeline, QueryPipeline};
use salva2d::rapier::prelude::vector;
use parry2d::na::Matrix;


mod conversion;


const SIM_SCALING_FACTOR: f32 = 0.02;
const PARTICLE_RAD: f32 = 0.1;

#[derive(NativeClass)]
#[inherit(Node)]
struct Physics {
    bodies: RigidBodySet,
    colliders: ColliderSet,
    physics_pipeline: PhysicsPipeline,
    island_manager: IslandManager,
    broad_phase: BroadPhase,
    narrow_phase: NarrowPhase,
    joint_set: JointSet,
    ccd_solver: CCDSolver,
    query_pipeline: QueryPipeline,
    fluids_pipeline: FluidsPipeline,
}

#[methods]
impl Physics {
    fn new(_owner: &Node) -> Self {
        Physics {
            bodies: RigidBodySet::new(),
            colliders: ColliderSet::new(),
            physics_pipeline: PhysicsPipeline::new(),
            island_manager: IslandManager::new(),
            broad_phase: BroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            joint_set: JointSet::new(),
            ccd_solver: CCDSolver::new(),
            query_pipeline: QueryPipeline::new(),
            fluids_pipeline: FluidsPipeline::new(PARTICLE_RAD, 2.0),
        }
    }

    #[export]
    fn add_rigid_body(
        &mut self,
        _owner: &Node,
        position: gdnative::core_types::Vector2,
        polygon: gdnative::core_types::Vector2Array,
        density: f32,
        restitution: f32,
        friction: f32,
        body_status: i32
    ) -> Vec<u32> {
        let mut status = RigidBodyType::Dynamic;
        if body_status == 1 {
            status = RigidBodyType::Static;
        } else if body_status == 2 {
            status = RigidBodyType::KinematicPositionBased;
        }
        let rb = RigidBodyBuilder::new(status).
            rotation(0.0).
            translation(Vector2::new(position.x * SIM_SCALING_FACTOR, position.y * SIM_SCALING_FACTOR)).
            ccd_enabled(body_status == 0).
            build();

        let rb_handle = self.bodies.insert(rb);

        // Build the collider.
        let collider = ColliderBuilder::convex_polyline(
            conversion::convert_to_points(polygon, SIM_SCALING_FACTOR)
        )
            .unwrap()
            .density(density)
            .restitution(restitution)
            .restitution_combine_rule(CoefficientCombineRule::Min)
            .friction(friction)
            .build();
        let geom_sample = salva2d::sampling::shape_surface_ray_sample(collider.shape(), PARTICLE_RAD).unwrap();

        let co_handle = self.colliders.insert_with_parent(collider, rb_handle, &mut self.bodies);
        let bo_handle = self.fluids_pipeline.liquid_world.add_boundary(Boundary::new(Vec::new()));
        self.fluids_pipeline.coupling.register_coupling(
            bo_handle,
            co_handle,
            ColliderSampling::DynamicContactSampling,
        );
        let (index, generation) = rb_handle.into_raw_parts();
        let (collider_index, collider_generation) = co_handle.into_raw_parts();
        let mut indices = Vec::new();
        indices.push(index);
        indices.push(collider_index);
        return indices;
    }

    #[export]
    fn deactivate_rigid_body(&mut self, _owner: &Node, collider_index: u32) {
        let collider_handle = ColliderHandle::from_raw_parts(collider_index, 0);
        let collider = self.colliders.get_mut(collider_handle).unwrap();
        collider.set_collision_groups(InteractionGroups::none());
    }

    #[export]
    fn activate_rigid_body(&mut self, _owner: &Node, collider_index: u32) {
        let collider_handle = ColliderHandle::from_raw_parts(collider_index, 0);
        let collider = self.colliders.get_mut(collider_handle).unwrap();
        collider.set_collision_groups(InteractionGroups::all());
    }

    #[export]
    fn deactivate_liquid_coupling(&mut self, _owner: &Node, collider_index: u32) {
        let boundary_handle = self.fluids_pipeline.coupling.unregister_coupling(ColliderHandle::from_raw_parts(collider_index, 0)).unwrap();
        self.fluids_pipeline.liquid_world.remove_boundary(boundary_handle);
    }

    #[export]
    fn activate_liquid_coupling(&mut self, _owner: &Node, collider_index: u32) {
        let bo_handle = self.fluids_pipeline.liquid_world.add_boundary(Boundary::new(Vec::new()));
        self.fluids_pipeline.coupling.register_coupling(
            bo_handle,
            ColliderHandle::from_raw_parts(collider_index, 0),
            ColliderSampling::DynamicContactSampling,
        );
    }

    #[export]
    fn add_sensor_to_body(&mut self, _owner: &Node, body_index: u32, position: gdnative::core_types::Vector2, polygon: gdnative::core_types::Vector2Array) -> u32 {
        let body_handle = RigidBodyHandle::from_raw_parts(body_index, 0);
        let sensor_collider = ColliderBuilder::convex_polyline(
            conversion::convert_to_points(polygon, SIM_SCALING_FACTOR)
        )
            .unwrap()
            .sensor(true)
            .build();
        let collider_handle = self.colliders.insert_with_parent(sensor_collider, body_handle, &mut self.bodies);
        let (collider_index, generation) = collider_handle.into_raw_parts();
        return collider_index;
    }

    #[export]
    fn add_sensor(&mut self, _owner: &Node, position: gdnative::core_types::Vector2, polygon: gdnative::core_types::Vector2Array) -> u32 {
        let rb = RigidBodyBuilder::new(RigidBodyType::KinematicPositionBased).
            //angular_inertia(0.0).
            rotation(0.0).
            translation(Vector2::new(position.x * SIM_SCALING_FACTOR, position.y * SIM_SCALING_FACTOR)).build();

        let body_handle = self.bodies.insert(rb);
        let sensor_collider = ColliderBuilder::convex_polyline(
            conversion::convert_to_points(polygon, SIM_SCALING_FACTOR)
        )
            .unwrap()
            .sensor(true)
            .build();
        let collider_handle = self.colliders.insert_with_parent(sensor_collider, body_handle, &mut self.bodies);
        let (collider_index, generation) = collider_handle.into_raw_parts();
        return collider_index;
    }

    #[export]
    fn get_contacting_colliders(&mut self, _owner: &Node, collider_index: u32) -> Vec<u32> {
        let mut collider_indices = Vec::new();
        let collider_handle = (ColliderHandle::from_raw_parts(collider_index, 0));
        let collider = self.colliders.get(collider_handle).unwrap();
        if collider.collision_groups() == InteractionGroups::none() {
            return collider_indices;
        }
        self.query_pipeline.intersections_with_shape(
            &self.colliders,
            collider.position(),
            collider.shape(),
            InteractionGroups::all(),
            None,
            |handle|
                {
                    let coll = &self.colliders.get(handle).unwrap();
                    if !coll.is_sensor() {
                        let (index, generation) = handle.into_raw_parts();
                        collider_indices.push(index);
                    }
                    true
                }
        );
        return collider_indices;
    }

    #[export]
    fn apply_force(&mut self, _owner: &Node, force: gdnative::core_types::Vector2, angular_force: f32, index: u32) {
        let body = self.bodies.get_mut(RigidBodyHandle::from_raw_parts(index, 0)).unwrap();
        body.apply_force(vector![force.x, force.y], true);
        body.apply_torque(angular_force, true);
    }

    #[export]
    fn set_velocity(&mut self, _owner: &Node, velocity: gdnative::core_types::Vector2, angular_force: f32, index: u32) {
        let mut body = self.bodies.get_mut(RigidBodyHandle::from_raw_parts(index, 0)).unwrap();
        body.set_linvel(Vector2::new(velocity.x, velocity.y), true);
        body.set_angvel(angular_force, true);
    }

    #[export]
    fn get_velocity(&mut self, _owner: &Node, index: u32) -> gdnative::core_types::Vector2 {
        let body = self.bodies.get_mut(RigidBodyHandle::from_raw_parts(index, 0)).unwrap();
        return gdnative::core_types::Vector2::new(body.linvel().x, body.linvel().y);
    }

    #[export]
    fn get_angular_velocity(&mut self, _owner: &Node, index: u32) -> f32 {
        let body = self.bodies.get_mut(RigidBodyHandle::from_raw_parts(index, 0)).unwrap();
        return body.angvel();
    }

    #[export]
    fn set_angular_velocity(&mut self, _owner: &Node, index: u32, angular_velocity: f32) {
        let body = self.bodies.get_mut(RigidBodyHandle::from_raw_parts(index, 0)).unwrap();
        return body.set_angvel(angular_velocity, true);
    }

    #[export]
    fn set_position(&mut self, _owner: &Node, position: gdnative::core_types::Vector2, angle: f32, index: u32) {
        let mut body = self.bodies.get_mut(RigidBodyHandle::from_raw_parts(index, 0)).unwrap();
        body.set_position(Isometry2::new(Vector2::new(position.x * SIM_SCALING_FACTOR, position.y * SIM_SCALING_FACTOR), angle), true);
    }

    #[export]
    fn get_position(&mut self, _owner: &Node, index: u32) -> gdnative::core_types::Vector2 {
        let body = self.bodies.get_mut(RigidBodyHandle::from_raw_parts(index, 0)).unwrap();
        return gdnative::core_types::Vector2::new(body.position().translation.x / SIM_SCALING_FACTOR, body.position().translation.y / SIM_SCALING_FACTOR);
    }

    #[export]
    fn get_rotation(&mut self, _owner: &Node, index: u32) -> f32 {
        let body = self.bodies.get_mut(RigidBodyHandle::from_raw_parts(index, 0)).unwrap();
        return body.position().rotation.angle();
    }

    #[export]
    fn set_angular_damping(&mut self, _owner: &Node, angular_damping: f32, index: u32) {
        let mut body = self.bodies.get_mut(RigidBodyHandle::from_raw_parts(index, 0)).unwrap();
        body.set_angular_damping(angular_damping);
    }

    #[export]
    fn set_angular_inertia(&mut self, _owner: &Node, angular_inertia: f32, index: u32) {
        let mut body = self.bodies.get_mut(RigidBodyHandle::from_raw_parts(index, 0)).unwrap();
        body.set_angular_damping(angular_inertia);
    }

    #[export]
    fn add_liquid(
        &mut self,
        _owner: &Node,
        droplets: gdnative::core_types::Vector2Array,
        velocities: gdnative::core_types::Vector2Array,
        fluid_viscosity_coefficent: f32,
        boundary_viscosity_coefficient: f32,
    ) -> gdnative::core_types::Vector2 {
        let mut points = conversion::convert_to_points(droplets, SIM_SCALING_FACTOR);

        let viscosity = ArtificialViscosity::new(fluid_viscosity_coefficent, boundary_viscosity_coefficient);
        let mut fluid = Fluid::new(points, PARTICLE_RAD, 1.0);
        fluid.velocities = conversion::convert_to_vec_of_vectors(velocities);
        fluid.nonpressure_forces.push(Box::new(viscosity.clone()));
        let fluid_handle: FluidHandle = self.fluids_pipeline.liquid_world.add_fluid(fluid);
        let idx: ContiguousArenaIndex = fluid_handle.into();
        let (fluid_index, generation) = idx.into_raw_parts();
        return gdnative::core_types::Vector2::new(fluid_index as f32, generation as f32);
    }

    #[export]
    fn remove_liquid(&mut self, _owner: &Node, liquid_index: gdnative::core_types::Vector2) {
        let fluid_handle = self.get_liquid_handle(liquid_index);
        self.fluids_pipeline.liquid_world.remove_fluid(fluid_handle);
    }

    #[export]
    fn get_liquid(&mut self, _owner: &Node) -> Vector2Array {
        let mut droplets = Vector2Array::new();
        for (i, fluid) in self.fluids_pipeline.liquid_world.fluids().iter() {
            for droplet in &fluid.positions {
                droplets.push(gdnative::core_types::Vector2::new(droplet.x / SIM_SCALING_FACTOR, droplet.y / SIM_SCALING_FACTOR));
            }
        }
        return droplets;
    }

    #[export]
    fn remove_particles(&mut self, _owner: &Node, fluid_index: gdnative::core_types::Vector2, collider_index: u32) {
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

        for (i, fluid) in self.fluids_pipeline.liquid_world.fluids().iter() {
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
        for (i, fluid) in self.fluids_pipeline.liquid_world.fluids().iter() {
            for velocity in &fluid.velocities {
                velocities.push(gdnative::core_types::Vector2::new(velocity.x / SIM_SCALING_FACTOR, velocity.y / SIM_SCALING_FACTOR));
            }
        }
        return velocities;
    }

    fn get_liquid_by_index(&self, liquid_index: gdnative::core_types::Vector2) -> &Fluid {
        let fluid_handle = self.get_liquid_handle(liquid_index);
        return self.fluids_pipeline.liquid_world.fluids().get(fluid_handle).unwrap();
    }

    fn get_mutable_liquid_by_index(&mut self, liquid_index: gdnative::core_types::Vector2) -> &mut Fluid {
        let fluid_handle = self.get_liquid_handle(liquid_index);
        return self.fluids_pipeline.liquid_world.fluids_mut().get_mut(fluid_handle).unwrap();
    }

    fn get_liquid_handle(&self, liquid_index: gdnative::core_types::Vector2) -> FluidHandle {
        let index = ContiguousArenaIndex::from_raw_parts(liquid_index.x as usize, liquid_index.y as u64);
        return FluidHandle::from(index);
    }

    #[export]
    fn get_contacting_liquids(&mut self, _owner: &Node, collider_index: u32) -> Vector2Array {
        let mut droplets = Vector2Array::new();
        let mut collider = self.colliders.get(ColliderHandle::from_raw_parts(collider_index, 0)).unwrap();
        if collider.collision_groups() == InteractionGroups::none() {
            return droplets;
        }
        let mut shape = collider.shape();
        let isometry = collider.position();

        for (i, fluid) in self.fluids_pipeline.liquid_world.fluids().iter() {
            for droplet in &fluid.positions {
                if shape.contains_point(isometry, droplet) {
                    droplets.push(gdnative::core_types::Vector2::new(droplet.x / SIM_SCALING_FACTOR, droplet.y / SIM_SCALING_FACTOR));
                }
            }
        }
        return droplets;
    }

    #[export]
    fn get_contacting_liquid_indices(&mut self, _owner: &Node, collider_index: u32) -> Vec<u32> {
        let mut droplet_indices = Vec::new();
        let mut collider = self.colliders.get(ColliderHandle::from_raw_parts(collider_index, 0)).unwrap();
        if collider.collision_groups() == InteractionGroups::none() {
            return droplet_indices;
        }
        let mut shape = collider.shape();
        let isometry = collider.position();

        for (i, fluid) in self.fluids_pipeline.liquid_world.fluids().iter() {
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

    #[export]
    fn _process(&mut self, _owner: &Node, delta: f32) {
        self.fluids_pipeline.liquid_world.step_with_coupling(
            1./60.,
            &vector![0.0, 9.81],
            &mut self.fluids_pipeline.coupling.as_manager_mut(&mut self.colliders, &mut self.bodies),
        );

        self.physics_pipeline.step(
            &vector![0.0, 9.81],
            &IntegrationParameters::default(),
            &mut self.island_manager,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.bodies,
            &mut self.colliders,
            &mut self.joint_set,
            &mut self.ccd_solver,
            &(),
            &(),
        );
        self.query_pipeline.update(&self.island_manager, &self.bodies, &self.colliders);
    }
}


fn init(handle: InitHandle) {
    handle.add_class::<Physics>();
}

godot_gdnative_init!();
godot_nativescript_init!(init);
godot_gdnative_terminate!();
