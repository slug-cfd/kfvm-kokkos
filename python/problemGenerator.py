import os
import shutil
import questionary

class ProblemSettings:
    def __init__(self):
        # Human readable choices for all questionary questions
        self.eqTypes = ["Euler","Ideal MHD-GLM","SR Hydro"]
        self.rkTypes = ["SSP(4,3)","SSP(10,4)","ThreeStarP","FourStarP"]
        self.rsTypes = {"Euler":["HLL","HLLC","Roe"],
                        "Ideal MHD-GLM":["KEPES","LLF"],
                        "SR Hydro":["HLL","LLF"]}
        self.floatPrecs = ["double","single"]
        self.execSpaces = ["Device","Host Parallel","Host Serial"]
        self.bcTypes = ["periodic","outflow","reflecting","user"]
        self.spaceDims = ["2","3"]
        self.stenRads = ["2","3"]
        self.numQuads = ["1","2","3","4","5"]

        # Dictionary to convert from human readable choices to code representation
        self.codeNames = {"Euler":"KFVM::EquationType::Hydro",
                          "Ideal MHD-GLM":"KFVM::EquationType::MHD_GLM",
                          "SR Hydro":"KFVM::EquationType::SRHydro",
                          "SSP(4,3)":"KFVM::RKType::SSP4_3_2",
                          "SSP(10,4)":"KFVM::RKType::SSP10_4_3",
                          "ThreeStarP":"KFVM::RKType::ThreeStarP",
                          "FourStarP":"KFVM::RKType::FourStarP",
                          "HLL":"KFVM::RSType::HLL",
                          "HLLC":"KFVM::RSType::HLLC",
                          "ROE":"KFVM::RSType::ROE",
                          "KEPES":"KFVM::RSType::MHD_GLM_KEPES",
                          "LLF":"KFVM::RSType::LLF",
                          "double":"1",
                          "single":"0",
                          "Device":"KFVM_EXEC_DEVICE",
                          "Host Parallel":"KFVM_EXEC_HOST",
                          "Host Serial":"KFVM_EXEC_SERIAL",
                          True:"true",
                          False:"false"}

    def query(self):
        self.kfvmDir = os.environ.get("KFVM_KOKKOS_ROOT")
        if self.kfvmDir == None:
            self.kfvmDir = questionary.path("Path to kfvm-kokkos:").ask()

        self.probName = questionary.text("Name of problem:").ask()
        self.probDir = questionary.path("Directory for problems (" + self.probName + " becomes subdirectory):",
                                        only_directories=True,default=os.getcwd()).ask()
        self.probPath = self.probDir + "/" + self.probName + "/"

        self.eqType = questionary.select("Equation type:",choices=self.eqTypes).ask()
        self.rsType = questionary.select("Riemann solver:",choices=self.rsTypes[self.eqType]).ask()

        self.rkType = questionary.select("Time integrator:",choices=self.rkTypes).ask()
        self.spaceDim = questionary.select("Space dimension:",choices=self.spaceDims).ask()
        self.stenRad = questionary.select("Stencil radius:",choices=self.stenRads).ask()
        self.numQuad = questionary.select("Number of quadrature points:",choices=self.numQuads,default="3").ask()

        self.bcWest = questionary.select("West BC type:",choices=self.bcTypes).ask()
        if self.bcWest == "periodic":
            self.bcEast = "periodic"
        else:
            self.bcEast = questionary.select("East BC type:",choices=self.bcTypes).ask()

        self.bcSouth = questionary.select("South BC type:",choices=self.bcTypes).ask()
        if self.bcSouth == "periodic":
            self.bcNorth = "periodic"
        else:
            self.bcNorth = questionary.select("North BC type:",choices=self.bcTypes).ask()

        self.bcBottom = questionary.select("Bottom BC type:",choices=self.bcTypes).ask()
        if self.bcBottom == "periodic":
            self.bcTop = "periodic"
        else:
            self.bcTop = questionary.select("Top BC type:",choices=self.bcTypes).ask()
        self.haveUserBCs = (self.bcWest == "user" or
                            self.bcEast == "user" or
                            self.bcSouth == "user" or
                            self.bcNorth == "user" or
                            self.bcBottom == "user" or
                            self.bcTop == "user")

        self.haveSrcTerm = questionary.confirm("Are there source terms?",default=False).ask()
        
        self.execSpace = questionary.select("Execution space:",choices=self.execSpaces).ask()
        
        self.floatPrec = questionary.select("Float precision:",choices=self.floatPrecs).ask()

    def verify(self):
        print()
        print("======== Configuration ========")
        print("Problem directory: ",self.probPath)
        print("Equation type: ",self.eqType)
        print("Riemann Solver: ",self.rsType)
        print("Time integrator: ",self.rkType)
        print("Space dimension: ",self.spaceDim)
        print("Stencil radius: ",self.stenRad)
        print("Number of quadrature points: ",self.numQuad)
        print("BC(West): ",self.bcWest)
        print("BC(East): ",self.bcEast)
        print("BC(South): ",self.bcSouth)
        print("BC(North): ",self.bcNorth)
        print("BC(Bottom): ",self.bcBottom)
        print("BC(Top): ",self.bcTop)
        print("Source terms present: ",self.haveSrcTerm)
        print("Execution space: ",self.execSpace)
        print("Float Precision: ",self.floatPrec)
        print()
        return questionary.confirm("Is this the right configuration?").ask()

    def generate(self):
        # Create directory for specific problem
        if not os.path.exists(self.probPath):
            os.mkdir(self.probPath)

        # Copy template files to new directory
        eqSuff = "Euler" if self.eqType == "Euler" else ("MHD_GLM" if self.eqType == "Ideal MHD-GLM" else "SRHydro")
        shutil.copy(self.kfvmDir + "/python/TmplFiles/InitialCondition_" + eqSuff + ".tmpl",
                    self.probPath + "InitialConditions.cpp")
        shutil.copy(self.kfvmDir + "/python/TmplFiles/SourceTerms_" + eqSuff + ".tmpl",
                    self.probPath + "SourceTerms.H")
        shutil.copy(self.kfvmDir + "/python/TmplFiles/UserBCs.tmpl",self.probPath + "UserBCs.H")
        initFile = self.probName.lower() + ".init"
        shutil.copy(self.kfvmDir + "/python/TmplFiles/init.tmpl",self.probPath + initFile)

        # Fill Definitions.H file
        with open(self.kfvmDir + "/python/TmplFiles/Definitions.tmpl","r") as infile:
            defn = (infile.read()
                    .replace("%{SPACE_DIM}",self.spaceDim)
                    .replace("%{STEN_RAD}",self.stenRad)
                    .replace("%{NUM_QUAD}",self.numQuad)
                    .replace("%{FLOAT_PREC}",self.codeNames[self.floatPrec])
                    .replace("%{EXEC_SPACE}",self.codeNames[self.execSpace])
                    .replace("%{EQ_TYPE}",self.codeNames[self.eqType])
                    .replace("%{RS_TYPE}",self.codeNames[self.rsType])
                    .replace("%{RK_TYPE}",self.codeNames[self.rkType]))
            with open(self.probPath + "Definitions.H","w") as outfile:
                outfile.write(defn)

        # Fill ProblemSetup.cpp file
        with open(self.kfvmDir + "/python/TmplFiles/ProblemSetup_" + eqSuff + ".tmpl","r") as f:
            probset = (f.read()
                       .replace("%{BC_WEST}",self.bcWest)
                       .replace("%{BC_EAST}",self.bcEast)
                       .replace("%{BC_SOUTH}",self.bcSouth)
                       .replace("%{BC_NORTH}",self.bcNorth)
                       .replace("%{BC_BOTTOM}",self.bcBottom)
                       .replace("%{BC_TOP}",self.bcTop)
                       .replace("%{HAVE_SOURCES}",self.codeNames[self.haveSrcTerm]))
            with open(self.probPath + "ProblemSetup.cpp","w") as outfile:
                outfile.write(probset)

        # Print out next instructions for user
        print("Next steps:")
        print("  Change to the new problem directory (cd " + self.probPath + ")")
        print("  Fill in InitialCondition.H")
        if self.haveSrcTerm:
            print("  Fill in SourceTerms.H")
        if self.haveUserBCs:
            print("  Fill in relevant fields of UserBCs.H")
        print("  Fill in " + initFile)
        print("  Run cmake command")
        print("  Run make -j")
        print("  Solve your problem!")

if __name__ == "__main__":
    while(True):
        ps = ProblemSettings()

        ps.query()

        goodConfig = ps.verify()
        if goodConfig:
            ps.generate()
            break

        redoConfig = questionary.confirm("Do you want to redo the configuration?").ask()
        if redoConfig:
            continue
        else:
            break
