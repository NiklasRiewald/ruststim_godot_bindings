extern crate nalgebra as na;

use gdnative::prelude::*;

use na::{Point2, Vector2, Isometry2};
use ncollide2d::shape::{ShapeHandle, Polyline};
use nphysics2d::force_generator::DefaultForceGeneratorSet;
use nphysics2d::joint::DefaultJointConstraintSet;
use nphysics2d::object::{BodyPartHandle, ColliderDesc, DefaultBodySet, DefaultColliderSet, RigidBodyDesc, BodyStatus, DefaultColliderHandle};
use nphysics2d::world::{DefaultGeometricalWorld, DefaultMechanicalWorld};
use salva2d::coupling::{ColliderCouplingSet};
use salva2d::object::{Fluid};
use salva2d::solver::{ArtificialViscosity, IISPHSolver};
use salva2d::LiquidWorld;

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
}

#[methods]
impl Physics {

    fn new(_owner: &Node) -> Self {
        Physics {
            bodies: DefaultBodySet::new(),
            colliders: DefaultColliderSet::new(),
            objects: Vec::new(),
            mechanical_world: DefaultMechanicalWorld::new(Vector2::new(0.0, -9.81)),
            geometrical_world: DefaultGeometricalWorld::new(),
            liquid_world: LiquidWorld::new(IISPHSolver::<f32>::new(), 0.1, 2.0),
            coupling_set: ColliderCouplingSet::new(),
            joint_constraints: DefaultJointConstraintSet::new(),
            force_generators: DefaultForceGeneratorSet::new(),
            particle_rad: 0.1
        }
    }

    #[export]
    fn add_rigid_body(&mut self, owner: &Node, position: gdnative::core_types::Vector2, polygon: gdnative::core_types::Vector2Array, is_static: bool) {
        let mut status = BodyStatus::Dynamic;
        if is_static {
            status = BodyStatus::Static;
        }
        let rb = RigidBodyDesc::new().status(status).
            translation(Vector2::new(position.x, position.y)).build();

        let rb_handle = self.bodies.insert(rb);

        // Build the collider.
        let geom = ShapeHandle::new(Self::convert_polygon(polygon));
        let co = ColliderDesc::new(geom)
            .density(1.0)
            .build(BodyPartHandle(rb_handle, 0));
        self.colliders.insert(co);
    }

    #[export]
    fn add_liquid(&mut self, owner: &Node, droplets: gdnative::core_types::Vector2Array) {
        let mut points = Vec::new();

        for drop in droplets.read().iter() {
            points.push(Point2::new(drop.x, drop.y));
        }

        let viscosity = ArtificialViscosity::new(0.5, 0.0);
        let mut fluid = Fluid::new(points, self.particle_rad, 1.0);
        fluid.nonpressure_forces.push(Box::new(viscosity.clone()));
        self.liquid_world.add_fluid(fluid);
    }

    #[export]
    fn get_liquid(&mut self, owner: &Node) -> Vector2Array {
        let mut droplets = Vector2Array::new();
        for (i, fluid) in  self.liquid_world.fluids().iter() {
            for droplet in &fluid.positions {
                droplets.push(gdnative::core_types::Vector2::new(droplet.x, droplet.y));
            }
        }
        return droplets;
    }

    #[export]
    fn get_polygons(&mut self, owner: &Node) -> Vector3Array {
        let mut polygons = Vector3Array::new();
        for (i, polygon) in self.colliders.iter() {
            let position: &Isometry2<f32> = collider.position();
            polygons.push(Vector3::new(position.translation.x, position.translation.y, position.rotation.angle()))
        }
        return polygons;
    }


    fn convert_polygon(polygon: gdnative::core_types::Vector2Array) -> Polyline<f32> {
        let mut points = Vec::new();
        for point in polygon.read().iter() {
            points.push(Point2::new(point.x, point.y));
        }
        return Polyline::new(points, None);
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

        #[cfg(feature = "fluids")]
            {
                if let Some(fluids) = &mut self.fluids {
                    let dt = self.mechanical_world.timestep();
                    let gravity = &self.mechanical_world.gravity;
                    self.liquid_world.step_with_coupling(
                        dt,
                        gravity,
                        &mut fluids
                            .coupling
                            .as_manager_mut(&self.colliders, &mut self.bodies),
                    );
                }
            }
    }
}


fn init(handle: InitHandle) {
    handle.add_class::<Physics>();
}

godot_gdnative_init!();
godot_nativescript_init!(init);
godot_gdnative_terminate!();
