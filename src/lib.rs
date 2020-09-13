extern crate nalgebra as na;

use gdnative::prelude::*;

use na::{Point2, Vector2, Isometry2};
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
    particle_rad: f32,
    sim_scaling_factor: f32,
}

#[methods]
impl Physics {

    fn new(_owner: &Node) -> Self {
        let sim_scaling_factor = 0.02;
        let rad = 5.0 * sim_scaling_factor;
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
            sim_scaling_factor,
        }
    }

    #[export]
    fn add_rigid_body(&mut self, owner: &Node, position: gdnative::core_types::Vector2, polygon: gdnative::core_types::Vector2Array, mass: f32, density: f32, body_status: i32) -> Vec<usize> {
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
            translation(Vector2::new(position.x * self.sim_scaling_factor, position.y * self.sim_scaling_factor)).build();

        let rb_handle = self.bodies.insert(rb);

        // Build the collider.
        let geom = ShapeHandle::new(self.convert_polygon2(polygon));
        let geom_sample =
            salva2d::sampling::shape_surface_ray_sample(&*geom, self.particle_rad).unwrap();
        let co = ColliderDesc::new(geom)
            //.margin(0.3)
            .density(density)
            .build(BodyPartHandle(rb_handle, 0));
        let co_handle = self.colliders.insert(co);
        let bo_handle = self.liquid_world.add_boundary(Boundary::new(Vec::new()));
        self.coupling_set.register_coupling(
            bo_handle,
            co_handle,
            CouplingMethod::DynamicContactSampling,
        );
        let (index, generation ) = rb_handle.into_raw_parts();
        let (collider_index, collider_generation ) = co_handle.into_raw_parts();
        let mut indices = Vec::new();
        indices.push(index);
        indices.push(collider_index);
        return indices;
    }

    #[export]
    fn deactivate_rigid_body(&mut self, owner: &Node, index: usize) {
        let body = self.bodies.get_mut(DefaultBodyHandle::from_raw_parts(index, 0)).unwrap();
        body.deactivate();
        body.set_status(BodyStatus::Disabled);
    }

    #[export]
    fn activate_rigid_body(&mut self, owner: &Node, index: usize) {
        let body = self.bodies.get_mut(DefaultBodyHandle::from_raw_parts(index, 0)).unwrap();
        body.activate();
        body.set_status(BodyStatus::Dynamic);
    }

    #[export]
    fn deactivate_liquid_coupling(&mut self, owner: &Node, collider_index: usize) {
        let boundary_handle = self.coupling_set.unregister_coupling(DefaultColliderHandle::from_raw_parts(collider_index, 0)).unwrap();
        self.liquid_world.remove_boundary(boundary_handle);
    }

    #[export]
    fn activate_liquid_coupling(&mut self, owner: &Node, collider_index: usize) {
        let bo_handle = self.liquid_world.add_boundary(Boundary::new(Vec::new()));
        self.coupling_set.register_coupling(
            bo_handle,
            DefaultBodyHandle::from_raw_parts(collider_index, 0),
            CouplingMethod::DynamicContactSampling,
        );
    }

    #[export]
    fn add_sensor_to_body(&mut self, owner: &Node, body_index: usize, position: gdnative::core_types::Vector2, polygon: gdnative::core_types::Vector2Array) -> usize {
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
    fn add_sensor(&mut self, owner: &Node, position: gdnative::core_types::Vector2, polygon: gdnative::core_types::Vector2Array) -> usize {
        let rb = RigidBodyDesc::new().status(BodyStatus::Kinematic).
            mass(1.0).
            angular_inertia(0.0).
            rotation(0.0).
            local_center_of_mass(Point2::new(0.0, 0.0)).
            translation(Vector2::new(position.x * self.sim_scaling_factor, position.y * self.sim_scaling_factor)).build();

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
    fn get_contacting_colliders(&mut self, owner: &Node, collider_index: usize) -> Vec<usize> {
        let mut collider_indices = Vec::new();
        for stuff in self.geometrical_world.colliders_in_proximity_of(&self.colliders,DefaultColliderHandle::from_raw_parts(collider_index, 0)).unwrap() {
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
    fn apply_force(&mut self, owner: &Node, force: gdnative::core_types::Vector2, angular_force: f32, index: usize) {
        let body = self.bodies.get_mut(DefaultBodyHandle::from_raw_parts(index, 0)).unwrap();
        body.apply_force(0, &Force2::new(Vector2::new(force.x, force.y), angular_force), ForceType::Force, true);
    }

    #[export]
    fn set_velocity(&mut self, owner: &Node, velocity: gdnative::core_types::Vector2, angular_force: f32, index: usize) {
        let body = self.bodies.rigid_body_mut(DefaultBodyHandle::from_raw_parts(index, 0)).unwrap();
        body.set_velocity(Velocity2::new(Vector2::new(velocity.x, velocity.y), angular_force));
    }

    #[export]
    fn get_velocity(&mut self, owner: &Node, index: usize) -> gdnative::core_types::Vector2 {
        let body = self.bodies.rigid_body_mut(DefaultBodyHandle::from_raw_parts(index, 0)).unwrap();
        return gdnative::core_types::Vector2::new(body.velocity().linear.x, body.velocity().linear.y);
    }

    #[export]
    fn get_angular_velocity(&mut self, owner: &Node, index: usize) -> f32 {
        let body = self.bodies.rigid_body_mut(DefaultBodyHandle::from_raw_parts(index, 0)).unwrap();
        return body.velocity().angular;
    }

    #[export]
    fn set_position(&mut self, owner: &Node, position: gdnative::core_types::Vector2, angle: f32, index: usize) {
        let body = self.bodies.rigid_body_mut(DefaultBodyHandle::from_raw_parts(index, 0)).unwrap();
        body.set_position(Isometry2::new(Vector2::new(position.x * self.sim_scaling_factor, position.y * self.sim_scaling_factor), angle));
    }

    #[export]
    fn get_position(&mut self, owner: &Node, index: usize) -> gdnative::core_types::Vector2 {
        let body = self.bodies.rigid_body_mut(DefaultBodyHandle::from_raw_parts(index, 0)).unwrap();
        return gdnative::core_types::Vector2::new(body.position().translation.x / self.sim_scaling_factor, body.position().translation.y / self.sim_scaling_factor);
    }

    #[export]
    fn get_rotation(&mut self, owner: &Node, index: usize) -> f32 {
        let body = self.bodies.rigid_body_mut(DefaultBodyHandle::from_raw_parts(index, 0)).unwrap();
        return body.position().rotation.angle();
    }

    #[export]
    fn set_mass(&mut self, owner: &Node, mass: f32, index: usize) {
        let body = self.bodies.rigid_body_mut(DefaultBodyHandle::from_raw_parts(index, 0)).unwrap();
        body.set_mass(mass);
    }

    #[export]
    fn set_angular_damping(&mut self, owner: &Node, angular_damping: f32, index: usize) {
        let body = self.bodies.rigid_body_mut(DefaultBodyHandle::from_raw_parts(index, 0)).unwrap();
        body.set_angular_damping(angular_damping);
    }

    #[export]
    fn set_angular_inertia(&mut self, owner: &Node, angular_inertia: f32, index: usize) {
        let body = self.bodies.rigid_body_mut(DefaultBodyHandle::from_raw_parts(index, 0)).unwrap();
        body.set_angular_inertia(angular_inertia);
    }

    #[export]
    fn add_liquid(&mut self, owner: &Node, droplets: gdnative::core_types::Vector2Array, velocities: gdnative::core_types::Vector2Array) -> gdnative::core_types::Vector2 {
        let mut points = self.convert_to_points(droplets);

        let viscosity = ArtificialViscosity::new(0.5, 0.0);
        let mut fluid = Fluid::new(points, self.particle_rad, 1.0);
        fluid.velocities = self.convert_to_vec_of_vectors(velocities);
        fluid.nonpressure_forces.push(Box::new(viscosity.clone()));
        let fluid_handle : FluidHandle = self.liquid_world.add_fluid(fluid);
        let idx: ContiguousArenaIndex =fluid_handle.into();
        let (fluid_index, generation ) = idx.into_raw_parts();
        return gdnative::core_types::Vector2::new(fluid_index as f32, generation as f32);
    }

    #[export]
    fn remove_liquid(&mut self, owner: &Node, liquid_index: gdnative::core_types::Vector2) {
        let fluid_handle = self.get_liquid_handle(liquid_index);
        self.liquid_world.remove_fluid(fluid_handle);
    }

    fn convert_to_vec_of_vectors(&mut self, vector: gdnative::core_types::Vector2Array) -> Vec<Vector2<f32>> {
        let mut rust_vec = Vec::new();
        for value in vector.read().iter() {
            rust_vec.push(Vector2::new(value.x, value.y));
        }
        return rust_vec;
    }

    #[export]
    fn get_liquid(&mut self, owner: &Node) -> Vector2Array {
        let mut droplets = Vector2Array::new();
        for (i, fluid) in self.liquid_world.fluids().iter() {
            for droplet in &fluid.positions {
                droplets.push(gdnative::core_types::Vector2::new(droplet.x / self.sim_scaling_factor, droplet.y / self.sim_scaling_factor));
            }
        }
        return droplets;
    }

    #[export]
    fn get_liquid_velocities(&mut self, owner: &Node, liquid_index: gdnative::core_types::Vector2) -> Vector2Array {
        let mut droplets = Vector2Array::new();
        for droplet in &self.get_liquid_by_index(liquid_index).velocities {
            droplets.push(gdnative::core_types::Vector2::new(droplet.x, droplet.y));
        }
        return droplets;
    }

    fn get_liquid_by_index(&mut self, liquid_index: gdnative::core_types::Vector2) -> &Fluid<f32> {
        let fluid_handle = self.get_liquid_handle(liquid_index);
        return self.liquid_world.fluids().get(fluid_handle).unwrap();
    }

    fn get_liquid_handle(&mut self, liquid_index: gdnative::core_types::Vector2) -> FluidHandle {
        let index = ContiguousArenaIndex::from_raw_parts(liquid_index.x as usize, liquid_index.y as u64);
        return FluidHandle::from(index);
    }

    #[export]
    fn get_contacting_liquids(&mut self, owner: &Node, collider_index: usize) -> Vector2Array  {
        let mut droplets = Vector2Array::new();
        let mut collider = self.colliders.get(DefaultColliderHandle::from_raw_parts(collider_index, 0)).unwrap();
        let mut shape = collider.shape();
        let isometry = collider.position();

        for (i, fluid) in  self.liquid_world.fluids().iter() {
            for droplet in &fluid.positions {
                if shape.contains_point(isometry, droplet) {
                    droplets.push(gdnative::core_types::Vector2::new(droplet.x / self.sim_scaling_factor, droplet.y / self.sim_scaling_factor));
                }
            }
        }
        return droplets;
    }

    #[export]
    fn get_polygons(&mut self, owner: &Node) -> Vector3Array {
        let mut polygons = Vector3Array::new();
        for (i, polygon) in self.colliders.iter() {
            let position: &Isometry2<f32> = polygon.position();
            polygons.push(Vector3::new(position.translation.x / self.sim_scaling_factor, position.translation.y / self.sim_scaling_factor, position.rotation.angle()))
        }
        return polygons;
    }

    fn convert_polygon(&mut self, polygon: gdnative::core_types::Vector2Array) -> Polyline<f32> {
        let mut points = self.convert_to_points(polygon);
        return Polyline::new(points, None);
    }

    fn convert_to_points(&mut self, godot_vector: gdnative::core_types::Vector2Array) -> Vec<Point2<f32>> {
        let mut points = Vec::new();
        for point in godot_vector.read().iter() {
            points.push(Point2::new(point.x * self.sim_scaling_factor, point.y* self.sim_scaling_factor));
        }
        return points;
    }

    fn convert_polygon2(&mut self, polygon: gdnative::core_types::Vector2Array) -> ConvexPolygon<f32> {
        let mut points = self.convert_to_points(polygon);
        return ConvexPolygon::try_from_points(&points).unwrap();
    }

    #[export]
    fn _process(&mut self, owner: &Node, delta: f32) {
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
