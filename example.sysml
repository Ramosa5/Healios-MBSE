package HEALIOS {
  public import ScalarValues::*;
  
  package MissionContext {

    // Actor/stakeholder parameter types must be part definitions (actor/stakeholder are part usages).
    part def Person;
    part def Organization;

    part def GroundOperator specializes Person;
    part def LaunchProvider specializes Organization;
    part def ScientificCommunity specializes Organization;

    // Environment can be used as an actor parameter (also a part usage), so define it as a part def.
    part def SpaceEnvironment;

    // Mission concept: 1U CubeSat in LEO running PSC degradation + self-healing experiment
    part def HealiosMission {
      // We will decompose ConOps and modes later
    }

  }

  package Requirements {

    // -------------------------
    // Mission-level requirements
    // -------------------------

    requirement def R_M_001_CharacterisePSC_IVCurves {
      subject mission : MissionContext::HealiosMission;

      actor environment : MissionContext::SpaceEnvironment;
      stakeholder operator : MissionContext::GroundOperator;
      stakeholder science : MissionContext::ScientificCommunity;
    }

    requirement def R_M_003_ReverseBiasSelfHealingInEclipse {
      subject mission : MissionContext::HealiosMission;

      actor environment : MissionContext::SpaceEnvironment;
      stakeholder operator : MissionContext::GroundOperator;
      stakeholder science : MissionContext::ScientificCommunity;

      attribute maxReverseBiasVoltage_V : Real;
    }

    requirement def R_M_006_1UPlatform {
      subject mission : MissionContext::HealiosMission;

      stakeholder operator : MissionContext::GroundOperator;

      attribute unitSize : String;
    }

    requirement def R_M_008_AutonomousExperimentExecution {
      subject mission : MissionContext::HealiosMission;

      actor environment : MissionContext::SpaceEnvironment;
      stakeholder operator : MissionContext::GroundOperator;
      stakeholder science : MissionContext::ScientificCommunity;
    }

    requirement def R_M_010_DownlinkTelemetryForCorrelation {
      subject mission : MissionContext::HealiosMission;

      stakeholder operator : MissionContext::GroundOperator;
      stakeholder science : MissionContext::ScientificCommunity;
    }

    // -------------------------
    // System-level (Space Segment)
    // -------------------------

    requirement def R_SS_001_AccommodatePayload {
      subject spacecraft : Structure::SpaceSegment;

      actor environment : MissionContext::SpaceEnvironment;
      stakeholder operator : MissionContext::GroundOperator;
      stakeholder science : MissionContext::ScientificCommunity;
    }

    requirement def R_SS_005_DetectEclipsePhase {
      subject spacecraft : Structure::SpaceSegment;

      actor environment : MissionContext::SpaceEnvironment;
      stakeholder operator : MissionContext::GroundOperator;
    }

    requirement def R_SS_006_SunPointingInIllumination {
      subject spacecraft : Structure::SpaceSegment;

      actor environment : MissionContext::SpaceEnvironment;
      stakeholder operator : MissionContext::GroundOperator;
    }

    requirement def R_SS_009_StoreExperimentDataBetweenContacts {
      subject spacecraft : Structure::SpaceSegment;

      stakeholder operator : MissionContext::GroundOperator;
      stakeholder science : MissionContext::ScientificCommunity;
    }

    requirement def R_SS_010_TxTelemetryCompatibleWithGroundProtocols {
      subject spacecraft : Structure::SpaceSegment;

      stakeholder operator : MissionContext::GroundOperator;
    }

    // -------------------------
    // System-level (Ground Segment)
    // -------------------------

    requirement def R_GS_009_BidirectionalComm {
      subject ground : Structure::GroundSegment;

      stakeholder operator : MissionContext::GroundOperator;
      stakeholder science : MissionContext::ScientificCommunity;
    }

    // -------------------------
    // Requirement usages (instantiated “shall” statements)
    // -------------------------

    package MissionReqts {

      requirement rq_M_001 : R_M_001_CharacterisePSC_IVCurves;

      requirement rq_M_003 : R_M_003_ReverseBiasSelfHealingInEclipse {
        :>> maxReverseBiasVoltage_V = 5.0;
      }

      requirement rq_M_006 : R_M_006_1UPlatform {
        :>> unitSize = "1U";
      }

      requirement rq_M_008 : R_M_008_AutonomousExperimentExecution;
      requirement rq_M_010 : R_M_010_DownlinkTelemetryForCorrelation;

    }

    package SpaceSegmentReqts {

      requirement rq_SS_001 : R_SS_001_AccommodatePayload;
      requirement rq_SS_005 : R_SS_005_DetectEclipsePhase;
      requirement rq_SS_006 : R_SS_006_SunPointingInIllumination;
      requirement rq_SS_009 : R_SS_009_StoreExperimentDataBetweenContacts;
      requirement rq_SS_010 : R_SS_010_TxTelemetryCompatibleWithGroundProtocols;

    }

    package GroundSegmentReqts {

      requirement rq_GS_009 : R_GS_009_BidirectionalComm;

    }

  }

  package Interfaces {

    // Keep interfaces abstract at first; we’ll refine signal sets/data products later.

    item def Telecommand;
    item def TelemetryPacket;
    item def ScienceDataBundle;

    port def SpaceToGroundPort {
      out item tm : TelemetryPacket;
      out item science : ScienceDataBundle;
      in item tc : Telecommand;
    }

    // Binary interface for the Space <-> Ground link
    interface def SpaceGroundLink {
      end space : SpaceToGroundPort;
      end ground : ~SpaceToGroundPort;
    }

  }

  package Structure {

    // -------------------------
    // Top-level system-of-interest
    // -------------------------

    part def HealiosSystem {
      part space : SpaceSegment;
      part ground : GroundSegment;

      interface commLink : Interfaces::SpaceGroundLink
        connect space.spaceGround to ground.spaceGround;
    }

    // -------------------------
    // Space Segment (1U CubeSat)
    // -------------------------

    part def SpaceSegment {

      port spaceGround : Interfaces::SpaceToGroundPort;

      part platform : CubeSatPlatform;
      part payload  : PerovskiteExperimentPayload;

      part eps   : ElectricalPowerSystem;
      part obdh  : OnBoardDataHandling;
      part comm  : CommunicationsSubsystem;
      part aocs  : AttitudeOrbitControlSystem;
      part tcs   : ThermalControlSystem;
      part str   : StructureMechanisms;
    }

    // -------------------------
    // Ground Segment
    // -------------------------

    part def GroundSegment {
      port spaceGround : Interfaces::SpaceToGroundPort;

      part groundStation  : GroundStation;
      part missionControl : MissionControlCentre;
      part dataProcessing : ScienceDataProcessing;
    }

    // -------------------------
    // Subsystem definitions (high-level placeholders)
    // -------------------------

    part def CubeSatPlatform;
    part def PerovskiteExperimentPayload;

    part def ElectricalPowerSystem;
    part def OnBoardDataHandling;
    part def CommunicationsSubsystem;
    part def AttitudeOrbitControlSystem;
    part def ThermalControlSystem;
    part def StructureMechanisms;

    part def GroundStation;
    part def MissionControlCentre;
    part def ScienceDataProcessing;

  }

  package Traceability {

    // Instantiate the system
    part healios : Structure::HealiosSystem;

    // ------------------------------------------------------------------
    // Satisfy requirement usages (correct SysML v2 form)
    // ------------------------------------------------------------------
    // Using the short satisfy form when referencing an existing requirement usage:
    // satisfy <qualified requirement usage> by <satisfying feature>;
    //
    // Mission requirements
   // satisfy Requirements::MissionReqts::rq_M_006 by healios.space.platform;
   // satisfy Requirements::MissionReqts::rq_M_001 by healios.space.payload;
   // satisfy Requirements::MissionReqts::rq_M_003 by healios.space.payload;
   // satisfy Requirements::MissionReqts::rq_M_008 by healios.space.obdh;
   // satisfy Requirements::MissionReqts::rq_M_010 by healios.space.comm;

    // Space segment requirements
    satisfy Requirements::SpaceSegmentReqts::rq_SS_001 by healios.space;
   // satisfy Requirements::SpaceSegmentReqts::rq_SS_005 by healios.space.aocs;
   // satisfy Requirements::SpaceSegmentReqts::rq_SS_006 by healios.space.aocs;
    //satisfy Requirements::SpaceSegmentReqts::rq_SS_009 by healios.space.obdh;
    //satisfy Requirements::SpaceSegmentReqts::rq_SS_010 by healios.space.comm;

    // Ground segment requirements
    //satisfy Requirements::GroundSegmentReqts::rq_GS_009 by healios.ground.groundStation;

    // ------------------------------------------------------------------
    // OPTIONAL alternative style (named satisfy requirement usage):
    // (kept here as commented template for later expansion)
    //
    // satisfy requirement sat_M_003 : Requirements::R_M_003_ReverseBiasSelfHealingInEclipse
    //   by healios.space.payload {
    //   :>> maxReverseBiasVoltage_V = 5.0;
    // }
    // ------------------------------------------------------------------

  }

}
