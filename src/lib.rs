extern crate nalgebra as na;

use godot::prelude::*;

use na::{Vector2, Isometry2};
use salva2d::object::{Fluid, Boundary, FluidHandle, ContiguousArenaIndex, ParticleId};
use salva2d::solver::ArtificialViscosity;
use std::cmp;
use godot::classes::Image;
use godot::classes::image::Format;
use salva2d::integrations::rapier::{FluidsPipeline, ColliderSampling};
use salva2d::rapier::dynamics::{RigidBodySet, IslandManager, IntegrationParameters, CCDSolver, RigidBodyType, CoefficientCombineRule, RigidBodyHandle, RigidBodyBuilder, ImpulseJointSet, MultibodyJointSet};
use salva2d::rapier::geometry::{ColliderSet, BroadPhase, NarrowPhase, ColliderBuilder, ColliderHandle, InteractionGroups};
use salva2d::rapier::pipeline::{PhysicsPipeline, QueryFilter, QueryPipeline};
use salva2d::rapier::prelude::vector;



mod conversion;

struct DimforgeGodotLibrary;

#[gdextension]
unsafe impl ExtensionLibrary for DimforgeGodotLibrary {}


const SIM_SCALING_FACTOR: f32 = 0.02;
const PARTICLE_RAD: f32 = 0.1;

use godot::classes::Node2D;

#[derive(GodotClass)]
#[class(no_init)]
struct RustPhysics {
    bodies: RigidBodySet,
    colliders: ColliderSet,
    physics_pipeline: PhysicsPipeline,
    island_manager: IslandManager,
    broad_phase: BroadPhase,
    narrow_phase: NarrowPhase,
    impulse_joints: ImpulseJointSet,
    multibody_joints: MultibodyJointSet,
    ccd_solver: CCDSolver,
    query_pipeline: QueryPipeline,
    fluids_pipeline: FluidsPipeline,
}

#[godot_api]
impl RustPhysics {

    fn create() -> Self {
        Self {
            bodies: RigidBodySet::new(),
            colliders: ColliderSet::new(),
            physics_pipeline: PhysicsPipeline::new(),
            island_manager: IslandManager::new(),
            broad_phase: BroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            impulse_joints: ImpulseJointSet::new(),
            multibody_joints: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),
            query_pipeline: QueryPipeline::new(),
            fluids_pipeline: FluidsPipeline::new(PARTICLE_RAD, 2.0),
        }
    }

    #[func]
    fn add_rigid_body(
        &mut self,
        position: godot::builtin::Vector2,
        polygon: godot::builtin::Array<godot::builtin::Vector2>,
        density: f32,
        restitution: f32,
        friction: f32,
        body_status: i32,
    ) -> VariantArray {
        let mut status = RigidBodyType::Dynamic;
        if body_status == 1 {
            status = RigidBodyType::Fixed;
        } else if body_status == 2 {
            status = RigidBodyType::KinematicVelocityBased;
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

        let co_handle = self.colliders.insert_with_parent(collider, rb_handle, &mut self.bodies);
        let bo_handle = self.fluids_pipeline.liquid_world.add_boundary(Boundary::new(Vec::new()));
        self.fluids_pipeline.coupling.register_coupling(
            bo_handle,
            co_handle,
            ColliderSampling::DynamicContactSampling,
        );
        let (index, generation) = rb_handle.into_raw_parts();
        let (collider_index, collider_generation) = co_handle.into_raw_parts();
        let mut indices = VariantArray::new();
        indices.push(Variant::from(godot::builtin::Vector2::new(index as f32, generation as f32)));
        indices.push(Variant::from(godot::builtin::Vector2::new(collider_index as f32, collider_generation as f32)));
        return indices;
    }

    #[func]
    fn deactivate_rigid_body(&mut self, collider_index: godot::builtin::Vector2) {
        let collider_handle = ColliderHandle::from_raw_parts(collider_index.x as u32, collider_index.y as u32);
        let collider = self.colliders.get_mut(collider_handle).unwrap();
        collider.set_collision_groups(InteractionGroups::none());
    }

    #[func]
    fn remove_rigid_body(&mut self, body_index: godot::builtin::Vector2) {
        let body_handle = RigidBodyHandle::from_raw_parts(body_index.x as u32, body_index.y as u32);
        self.bodies.remove(
            body_handle,
            &mut self.island_manager,
            &mut self.colliders,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            true
        );
    }

    #[func]
    fn activate_rigid_body(&mut self, collider_index: godot::builtin::Vector2) {
        let collider_handle = ColliderHandle::from_raw_parts(collider_index.x as u32, collider_index.y as u32);
        let collider = self.colliders.get_mut(collider_handle).unwrap();
        collider.set_collision_groups(InteractionGroups::all());
    }

    #[func]
    fn deactivate_liquid_coupling(&mut self, collider_index: godot::builtin::Vector2) {
        let boundary_handle = self.fluids_pipeline.coupling.unregister_coupling(ColliderHandle::from_raw_parts(collider_index.x as u32, collider_index.y as u32)).unwrap();
        self.fluids_pipeline.liquid_world.remove_boundary(boundary_handle);
    }

    #[func]
    fn activate_liquid_coupling(&mut self, collider_index: godot::builtin::Vector2) {
        let bo_handle = self.fluids_pipeline.liquid_world.add_boundary(Boundary::new(Vec::new()));
        self.fluids_pipeline.coupling.register_coupling(
            bo_handle,
            ColliderHandle::from_raw_parts(collider_index.x as u32, collider_index.y as u32),
            ColliderSampling::DynamicContactSampling,
        );
    }

    #[func]
    fn add_sensor_to_body(&mut self, body_index: godot::builtin::Vector2, position: godot::builtin::Vector2, polygon: godot::builtin::Array<godot::builtin::Vector2>) -> godot::builtin::Vector2 {
        let body_handle = RigidBodyHandle::from_raw_parts(body_index.x as u32, body_index.y as u32);
        let body = self.bodies.get_mut(body_handle).unwrap();
        let body_translation = body.position().translation;
        let sensor_collider = ColliderBuilder::convex_polyline(
            conversion::convert_to_points(polygon, SIM_SCALING_FACTOR)
        )
            .unwrap()
            .position(
                Isometry2::new(
                    Vector2::new(
                        position.x * SIM_SCALING_FACTOR - body_translation.x,
                        position.y * SIM_SCALING_FACTOR - body_translation.y,
                    ),
                    0.0,
                )
            )
            .sensor(true)
            .build();
        let collider_handle = self.colliders.insert_with_parent(sensor_collider, body_handle, &mut self.bodies);
        let (collider_index, generation) = collider_handle.into_raw_parts();
        return godot::builtin::Vector2::new(collider_index as f32, generation as f32);
    }

    #[func]
    fn add_sensor(&mut self, position: godot::builtin::Vector2, polygon: godot::builtin::Array<godot::builtin::Vector2>) -> godot::builtin::Vector2 {
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
        return godot::builtin::Vector2::new(collider_index as f32, generation as f32);
    }

    #[func]
    fn get_contacting_colliders(&mut self, collider_index: godot::builtin::Vector2) -> VariantArray {
        let mut collider_indices = VariantArray::new();
        let collider_handle = ColliderHandle::from_raw_parts(collider_index.x as u32, collider_index.y as u32);
        let collider = self.colliders.get(collider_handle).unwrap();
        if collider.collision_groups() == InteractionGroups::none() {
            return collider_indices;
        }
        self.query_pipeline.intersections_with_shape(
            &self.bodies,
            &self.colliders,
            collider.position(),
            collider.shape(),
            QueryFilter::default(),
            |handle|
                {
                    let coll = &self.colliders.get(handle).unwrap();
                    if !coll.is_sensor() {
                        let (index, generation) = handle.into_raw_parts();
                        collider_indices.push(Variant::from(godot::builtin::Vector2::new(index as f32, generation as f32)));
                    }
                    true
                },
        );
        return collider_indices;
    }

    #[func]
    fn apply_force(&mut self, force: godot::builtin::Vector2, angular_force: f32, index: godot::builtin::Vector2) {
        let body = self.bodies.get_mut(RigidBodyHandle::from_raw_parts(index.x as u32, index.y as u32)).unwrap();
        body.add_force(vector![force.x, force.y], true);
        body.apply_torque_impulse(angular_force, true);
    }

    #[func]
    fn set_velocity(&mut self, velocity: godot::builtin::Vector2, angular_force: f32, index: godot::builtin::Vector2) {
        let body = self.bodies.get_mut(RigidBodyHandle::from_raw_parts(index.x as u32, index.y as u32)).unwrap();
        body.set_linvel(Vector2::new(velocity.x, velocity.y), true);
        body.set_angvel(angular_force, true);
    }

    #[func]
    fn get_velocity(&mut self, index: godot::builtin::Vector2) -> godot::builtin::Vector2 {
        let body = self.bodies.get_mut(RigidBodyHandle::from_raw_parts(index.x as u32, index.y as u32)).unwrap();
        return godot::builtin::Vector2::new(body.linvel().x, body.linvel().y);
    }

    #[func]
    fn get_angular_velocity(&mut self, index: godot::builtin::Vector2) -> f32 {
        let body = self.bodies.get_mut(RigidBodyHandle::from_raw_parts(index.x as u32, index.y as u32)).unwrap();
        return body.angvel();
    }

    #[func]
    fn set_angular_velocity(&mut self, index: godot::builtin::Vector2, angular_velocity: f32) {
        let body = self.bodies.get_mut(RigidBodyHandle::from_raw_parts(index.x as u32, index.y as u32)).unwrap();
        return body.set_angvel(angular_velocity, true);
    }

    #[func]
    fn set_position(&mut self, position: godot::builtin::Vector2, angle: f32, index: godot::builtin::Vector2) {
        let body = self.bodies.get_mut(RigidBodyHandle::from_raw_parts(index.x as u32, index.y as u32)).unwrap();
        body.set_position(Isometry2::new(Vector2::new(position.x * SIM_SCALING_FACTOR, position.y * SIM_SCALING_FACTOR), angle), true);
    }

    #[func]
    fn get_position(&mut self, index: godot::builtin::Vector2) -> godot::builtin::Vector2 {
        let body = self.bodies.get_mut(RigidBodyHandle::from_raw_parts(index.x as u32, index.y as u32)).unwrap();
        return godot::builtin::Vector2::new(body.position().translation.x / SIM_SCALING_FACTOR, body.position().translation.y / SIM_SCALING_FACTOR);
    }

    #[func]
    fn get_rotation(&mut self, index: godot::builtin::Vector2) -> f32 {
        let body = self.bodies.get_mut(RigidBodyHandle::from_raw_parts(index.x as u32, index.y as u32)).unwrap();
        return body.position().rotation.angle();
    }

    #[func]
    fn set_angular_damping(&mut self, angular_damping: f32, index: godot::builtin::Vector2) {
        let body = self.bodies.get_mut(RigidBodyHandle::from_raw_parts(index.x as u32, index.y as u32)).unwrap();
        body.set_angular_damping(angular_damping);
    }

    #[func]
    fn set_angular_inertia(&mut self, angular_inertia: f32, index: godot::builtin::Vector2) {
        let body = self.bodies.get_mut(RigidBodyHandle::from_raw_parts(index.x as u32, index.y as u32)).unwrap();
        body.set_angular_damping(angular_inertia);
    }

    #[func]
    fn add_liquid(
        &mut self,
        droplets: godot::builtin::Array<godot::builtin::Vector2>,
        velocities: godot::builtin::Array<godot::builtin::Vector2>,
        accelerations: godot::builtin::Array<godot::builtin::Vector2>,
        fluid_viscosity_coefficent: f32,
        boundary_viscosity_coefficient: f32,
    ) -> godot::builtin::Vector2 {
        let points = conversion::convert_to_points(droplets, SIM_SCALING_FACTOR);

        let viscosity = ArtificialViscosity::new(fluid_viscosity_coefficent, boundary_viscosity_coefficient);
        let mut fluid = Fluid::new(points, PARTICLE_RAD, 1.0);
        fluid.velocities = conversion::convert_to_vec_of_vectors(velocities);
        fluid.accelerations = conversion::convert_to_vec_of_vectors(accelerations);
        fluid.nonpressure_forces.push(Box::new(viscosity.clone()));
        let fluid_handle: FluidHandle = self.fluids_pipeline.liquid_world.add_fluid(fluid);
        let idx: ContiguousArenaIndex = fluid_handle.into();
        let (fluid_index, generation) = idx.into_raw_parts();
        return godot::builtin::Vector2::new(fluid_index as f32, generation as f32);
    }

    #[func]
    fn remove_liquid(&mut self, liquid_index: godot::builtin::Vector2) {
        let fluid_handle = self.get_liquid_handle(liquid_index);
        self.fluids_pipeline.liquid_world.remove_fluid(fluid_handle);
    }

    #[func]
    fn get_liquid(&mut self) -> Array<godot::builtin::Vector2> {
        let mut droplets = Array::new();
        for (_i, fluid) in self.fluids_pipeline.liquid_world.fluids().iter() {
            for droplet in &fluid.positions {
                droplets.push(godot::builtin::Vector2::new(droplet.x / SIM_SCALING_FACTOR, droplet.y / SIM_SCALING_FACTOR));
            }
        }
        return droplets;
    }

    #[func]
    fn remove_particles(&mut self, fluid_index: godot::builtin::Vector2, collider_index: godot::builtin::Vector2) {
        let particle_indices = self.get_contacting_liquid_indices(collider_index);
        let fluid = self.get_mutable_liquid_by_index(fluid_index);

        for particle_index in particle_indices.iter_shared() {
            fluid.delete_particle_at_next_timestep(particle_index.to::<i64>() as usize);
        }
    }

    #[func]
    fn add_particles(&mut self, fluid_index: godot::builtin::Vector2, new_positions: Array<godot::builtin::Vector2>, new_velocities: Array<godot::builtin::Vector2>) {
        let fluid = self.get_mutable_liquid_by_index(fluid_index);
        let positions = conversion::convert_to_points(new_positions, SIM_SCALING_FACTOR);
        let velocities = conversion::convert_to_vec_of_vectors(new_velocities);
        fluid.add_particles(&positions[..], Some(&velocities[..]));
    }

    #[func]
    fn get_liquid_raster(&mut self, x_min: f32, x_max: f32, y_min: f32, y_max: f32, resolution: f32, meta_ball_influence: i64) -> Gd < Image > {
        let width = ((x_max - x_min) / resolution + 1.0).ceil() as i64;
        let height = ((y_max - y_min) / resolution + 1.0).ceil() as i64;
        let mut data = vec![vec![0.0f32; height as usize]; width as usize];
        //let mut velocities = vec![vec![0.0f32; height as usize]; width as usize];


        let mut image = Image::create(width as i32, height as i32, false, Format::RGBA8).unwrap();
        image.fill(Color::from_rgba(0., 0.0, 0.0, 0.0));
        let scaled_x_min = x_min * SIM_SCALING_FACTOR;
        let scaled_x_max = x_max * SIM_SCALING_FACTOR;
        let scaled_y_min = y_min * SIM_SCALING_FACTOR;
        let scaled_y_max = y_max * SIM_SCALING_FACTOR;

        for (_i, fluid) in self.fluids_pipeline.liquid_world.fluids().iter() {
            for droplet in &fluid.positions {
                if droplet.x >= scaled_x_min && droplet.x <= scaled_x_max && droplet.y >= scaled_y_min && droplet.y <= scaled_y_max {
                    let true_x = (droplet.x / SIM_SCALING_FACTOR - x_min) / resolution;
                    let x = true_x.round() as i64;
                    let x_start = cmp::max(0, x - meta_ball_influence);
                    let x_end = cmp::min(width, x + meta_ball_influence);

                    let true_y = (droplet.y / SIM_SCALING_FACTOR - y_min) / resolution;
                    let y = true_y.round() as i64;
                    let y_start = cmp::max(0, y - meta_ball_influence);
                    let y_end = cmp::min(height, y + meta_ball_influence);

                    for x_i in x_start..x_end {
                        for y_i in y_start..y_end {
                            //original
                            //data[x_i as usize][y_i as usize] += 1. / ((true_x - x_i as f32).powf(2.0) + (true_y - y_i as f32).powf(2.0));
                            //original / 4
                            let distance = (1. / ((true_x - x_i as f32).powf(2.0) + (true_y - y_i as f32).powf(2.0))).min(1.0) / 2.;
                            data[x_i as usize][y_i as usize] += distance;
//                            velocities[x_i as usize][y_i as usize] = velocities[x_i as usize][y_i as usize].max(
//                                distance * (&fluid.velocities[pos].x.powf(2.0) + &fluid.velocities[pos].y.powf(2.0)).powf(0.5)
//                            );
                        }
                    }
                }
            }
        }

        for x_i in 0..width {
            for y_i in 0..height {
                image.set_pixel(
                    x_i as i32,
                    y_i as i32,
                    Color::from_rgba(
                        //velocities[x_i as usize][y_i as usize],
                        0.0 as f32,
                        0.0 as f32,
                        0.0 as f32,
                        data[x_i as usize][y_i as usize],
                    ),
                );
            }
        }

        return image;
    }

    #[func]
    fn get_liquid_velocities(&mut self, liquid_index: godot::builtin::Vector2) -> Array<godot::builtin::Vector2> {
        let mut droplets = Array::new();
        for droplet in &self.get_liquid_by_index(liquid_index).velocities {
            droplets.push(godot::builtin::Vector2::new(droplet.x, droplet.y));
        }
        return droplets;
    }

    #[func]
    fn get_all_liquid_velocities(&mut self) -> Array<godot::builtin::Vector2> {
        let mut velocities = Array::new();
        for (_i, fluid) in self.fluids_pipeline.liquid_world.fluids().iter() {
            for velocity in &fluid.velocities {
                velocities.push(godot::builtin::Vector2::new(velocity.x, velocity.y));
            }
        }
        return velocities;
    }

    #[func]
    fn get_all_liquid_accelerations(&mut self) -> Array<godot::builtin::Vector2> {
        let mut accelerations = Array::new();
        for (_i, fluid) in self.fluids_pipeline.liquid_world.fluids().iter() {
            for acceleration in &fluid.accelerations {
                accelerations.push(godot::builtin::Vector2::new(acceleration.x, acceleration.y));
            }
        }
        return accelerations;
    }

    fn get_liquid_by_index(&self, liquid_index: godot::builtin::Vector2) -> &Fluid {
        let fluid_handle = self.get_liquid_handle(liquid_index);
        return self.fluids_pipeline.liquid_world.fluids().get(fluid_handle).unwrap();
    }

    fn get_mutable_liquid_by_index(&mut self, liquid_index: godot::builtin::Vector2) -> &mut Fluid {
        let fluid_handle = self.get_liquid_handle(liquid_index);
        return self.fluids_pipeline.liquid_world.fluids_mut().get_mut(fluid_handle).unwrap();
    }

    fn get_liquid_handle(&self, liquid_index: godot::builtin::Vector2) -> FluidHandle {
        let index = ContiguousArenaIndex::from_raw_parts(liquid_index.x as usize, liquid_index.y as u64);
        return FluidHandle::from(index);
    }

    #[func]
    fn get_contacting_liquids(&mut self, collider_index: godot::builtin::Vector2) -> Array<godot::builtin::Vector2> {
        let mut droplets = Array::new();
        let collider = self.colliders.get(ColliderHandle::from_raw_parts(collider_index.x as u32, collider_index.y as u32)).unwrap();
        if collider.collision_groups() == InteractionGroups::none() {
            return droplets;
        }
        let shape = collider.shape();
        let isometry = collider.position();

        for (_i, fluid) in self.fluids_pipeline.liquid_world.fluids().iter() {
            for droplet in &fluid.positions {
                if shape.contains_point(isometry, droplet) {
                    droplets.push(godot::builtin::Vector2::new(droplet.x / SIM_SCALING_FACTOR, droplet.y / SIM_SCALING_FACTOR));
                }
            }
        }
        return droplets;
    }

    #[func]
    fn get_contacting_liquid_indices(&mut self, collider_index: godot::builtin::Vector2) -> VariantArray {
        let droplet_indices = VariantArray::new();
        let collider = self.colliders.get(ColliderHandle::from_raw_parts(collider_index.x as u32, collider_index.y as u32)).unwrap();
        if collider.collision_groups() == InteractionGroups::none() {
            return droplet_indices;
        }
        let shape = collider.shape();
        let isometry = collider.position();

        let mut liquid_indices = VariantArray::new();

        for index in self.fluids_pipeline.liquid_world.particles_intersecting_shape(isometry, shape).filter_map(move |entry| match entry {
            ParticleId::FluidParticle(_fid, pid) => {
                Some(pid as u32)
            }
            ParticleId::BoundaryParticle(_bid, pid) => {
                Some(pid as u32)
            }
        }) {
            liquid_indices.push(Variant::from(index))
        }

        return liquid_indices;
    }

    #[func]
    fn get_polygons(&mut self) -> Dictionary {
        let mut polygons = Dictionary::new();
        for (handle, polygon) in self.colliders.iter() {
            let position: &Isometry2<f32> = polygon.position();
            let (collider_index, generation) = handle.into_raw_parts();
            polygons.insert(
                collider_index.to_string() + "_" + &generation.to_string(),
                Vector3::new(position.translation.x / SIM_SCALING_FACTOR, position.translation.y / SIM_SCALING_FACTOR, position.rotation.angle()),
            );
        }
        return polygons;
    }

    fn process(&mut self, delta: f64) {
        self.fluids_pipeline.liquid_world.step_with_coupling(
            1. / 60.,
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
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            &mut self.ccd_solver,
            Some(&mut self.query_pipeline),
            &(),
            &(),
        );
        self.query_pipeline.update(&self.bodies, &self.colliders);
    }
}

